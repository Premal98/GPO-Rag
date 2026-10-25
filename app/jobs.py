"""Background processing: upload a PDF once and it is parsed, chunked and ingested.

POST /documents saves and checks the file inside the request (fast), then
queues the slow work - Docling parsing, Gemini enrichment, embedding - and
returns immediately. Progress is stored in the documents table, so
GET /documents/{id} shows it:

  uploaded -> parsing -> enriching -> embedding -> ready   (or failed)

One worker thread runs the jobs one at a time: Docling uses a lot of memory
and CPU, and the demo never needs two documents processed at once.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from app import config
from app.document_registry import DocumentRegistryError, ready_store
from app.embeddings import is_embedding_configured
from app.ingestion import IngestionError, ingest_document
from app.pdf_ingestion import PDFIngestionError, open_pdf
from app.pipeline import process_saved_pdf, save_pdf_upload

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document-job")

# Documents queued or being processed by this server process. The database
# status alone can be stale (e.g. if recording "failed" itself failed).
_active = set()
_active_lock = threading.Lock()

INTERRUPTED_MESSAGE = "The server restarted while this document was being processed. Upload it again."


def is_active(document_id):
    """True while this server is processing (or about to process) the document."""
    with _active_lock:
        return document_id in _active


def _record_failure(set_status, document_id, message):
    """Mark the job failed; retry once, and log if the database is unreachable."""
    for attempt in (1, 2):
        try:
            set_status("failed", message)
            return
        except Exception:
            if attempt == 2:
                logger.exception("Could not mark document %s as failed", document_id)
            else:
                time.sleep(2)


def run_document_job(document_id, saved_path, filename, profile, force, store, upload_dir,
                     ingest=ingest_document):
    """Process and ingest one document, recording each stage in the documents table.

    ingest can be replaced in tests.
    """
    def set_status(status, message=None, new_profile=None):
        store.set_document_status(document_id, filename, status, message, new_profile or profile)

    try:
        result = process_saved_pdf(
            saved_path, document_id, filename, upload_dir, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
            profile=profile, reuse_result=not force, on_status=set_status,
        )
        profile = result["profile"]
        if not result["chunks"]:
            _record_failure(set_status, document_id, result["message"])
            return
        set_status("embedding")
        # Ingests the result computed here (not whatever is on disk by now),
        # replacing any chunks stored earlier, and marks the document ready
        ingest(document_id, upload_dir, force=True, store=store, processed=result)
    except (PDFIngestionError, IngestionError) as err:
        _record_failure(set_status, document_id, err.message)
    except Exception as err:  # keep the worker alive and the failure visible
        logger.exception("Processing document %s failed", document_id)
        _record_failure(set_status, document_id, f"Processing failed: {err}")
    finally:
        with _active_lock:
            _active.discard(document_id)


def submit_document(file_obj, filename, content_type, profile=None, force=False, store=None,
                    upload_dir=None, run=None):
    """Save an upload and queue its processing. Returns a dict for the response.

    Its "accepted" key is False when nothing was queued: the document is
    already ready (use force=True to process it again) or already queued.
    run(job) executes the job; it defaults to the background worker.
    """
    if not is_embedding_configured():
        raise DocumentRegistryError("GOOGLE_API_KEY is not set in .env.", 503)
    store = ready_store(store)
    upload_dir = Path(upload_dir or config.UPLOAD_DIR)

    saved_path, document_id, filename = save_pdf_upload(
        file_obj, filename, content_type, upload_dir, int(config.MAX_UPLOAD_MB * 1024 * 1024))
    try:
        open_pdf(saved_path)  # reject corrupted or password-protected files now, not later
    except PDFIngestionError:
        saved_path.unlink(missing_ok=True)
        raise

    response = {"document_id": document_id, "filename": filename,
                "status_url": f"/documents/{document_id}"}
    with _active_lock:
        if document_id in _active:
            return {**response, "accepted": False, "status": "processing",
                    "message": "This document is already being processed."}
        try:
            existing = store.status(document_id)["document"]
        except psycopg.Error as err:
            raise DocumentRegistryError(f"Database error: {str(err).strip()}", 503) from err
        # A busy status without a running job here is stale; it may be processed again
        if existing and existing["status"] == "ready" and not force \
                and profile in (None, existing["profile"]):
            return {**response, "accepted": False, "status": "ready",
                    "profile": existing["profile"],
                    "message": "This document is already processed and ready. Use force=true "
                               "to process it again."}
        try:
            store.set_document_status(document_id, filename, "uploaded",
                                      "Waiting to be processed.", profile)
        except psycopg.Error as err:
            raise DocumentRegistryError(f"Database error: {str(err).strip()}", 503) from err
        _active.add(document_id)

    job = (document_id, saved_path, filename, profile, force, store, upload_dir)
    try:
        (run or (lambda args: _executor.submit(run_document_job, *args)))(job)
    except Exception as err:
        with _active_lock:
            _active.discard(document_id)
        _record_failure(lambda s, m: store.set_document_status(document_id, filename, s, m),
                        document_id, f"Could not start processing: {err}")
        raise DocumentRegistryError(f"Could not start processing: {err}", 503) from err
    return {**response, "accepted": True, "status": "uploaded",
            "message": "Processing started. Check status_url until status is ready."}


def recover_interrupted(store=None):
    """On startup, mark documents a previous server run left unfinished as failed."""
    try:
        return ready_store(store).fail_unfinished(INTERRUPTED_MESSAGE)
    except Exception:  # e.g. no database configured yet: nothing to recover
        logger.info("Skipped recovering unfinished documents.", exc_info=True)
        return 0
