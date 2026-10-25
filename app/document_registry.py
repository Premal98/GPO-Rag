"""Lists and deletes documents (the chatbots a frontend can offer).

A document appears here once it has been ingested. Its document_id is what
POST /chat needs to answer from that document only.
"""

import shutil
from pathlib import Path

import psycopg

from app.database import MIGRATION_002, is_database_configured
from app.document_store import DOCUMENT_ID_PATTERN, load_result
from app.vector_store import ChunkStore


class DocumentRegistryError(Exception):
    """A problem we report back to the user, with an HTTP status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def ready_store(store):
    """Return a store whose database has both SQL scripts applied."""
    if store is None:
        if not is_database_configured():
            raise DocumentRegistryError("SUPABASE_DATABASE_URL is not set in .env.", 503)
        store = ChunkStore()
    try:
        setup = store.status()
    except psycopg.Error as err:
        raise DocumentRegistryError(f"Database error: {str(err).strip()}", 503) from err
    if setup["embedding_dimension"] is None:
        raise DocumentRegistryError(
            "The document_chunks table does not exist. Run sql/001_create_document_chunks.sql.", 503
        )
    if not setup["migrated"]:
        raise DocumentRegistryError(f"The documents table is missing. Run {MIGRATION_002}.", 503)
    return store


def list_documents(store=None):
    """Return {"documents": [...], "count": n} for every ingested document."""
    store = ready_store(store)
    try:
        documents = store.list_documents()
    except psycopg.Error as err:
        raise DocumentRegistryError(f"Database error: {str(err).strip()}", 503) from err
    return {"documents": documents, "count": len(documents)}


PROCESSING_KEYS = ("parser", "profile_reason", "layout_summary", "warnings", "page_count",
                   "pages_needing_ocr", "chunk_count", "chunk_settings")


def get_document(document_id, upload_dir, store=None):
    """One document's registry entry (with its processing status) and processing details.

    "processing" holds what the last parse found (parser, layout summary,
    warnings, ...). A document that was uploaded with POST /documents/upload
    but not ingested has status "not_ingested".
    """
    if not DOCUMENT_ID_PATTERN.match(document_id):
        raise DocumentRegistryError("Document not found.", 404)
    store = ready_store(store)
    try:
        registered = store.status(document_id)["document"]
    except psycopg.Error as err:
        raise DocumentRegistryError(f"Database error: {str(err).strip()}", 503) from err
    processed = load_result(upload_dir, document_id)
    if registered is None and processed is None:
        raise DocumentRegistryError("Document not found.", 404)

    document = registered or {"document_id": document_id, "filename": processed["filename"],
                              "profile": processed.get("profile"), "status": "not_ingested",
                              "status_message": "Call POST /documents/{id}/ingest to make it chat-ready."}
    details = {key: processed.get(key) for key in PROCESSING_KEYS} if processed else None
    return {**document, "chat_ready": document["status"] == "ready", "processing": details}


def _local_files(upload_dir, document_id):
    upload_dir = Path(upload_dir)
    files = list(upload_dir.glob(f"{document_id}_*.pdf"))
    processed = upload_dir / "processed" / f"{document_id}.json"
    if processed.exists():
        files.append(processed)
    return files


def delete_document(document_id, upload_dir, store=None):
    """Delete a document's chunks, its registry row, and its uploaded files."""
    if not DOCUMENT_ID_PATTERN.match(document_id):
        raise DocumentRegistryError("Document not found.", 404)  # also blocks paths like "../x"
    store = ready_store(store)
    try:
        deleted = store.delete_document(document_id)
    except psycopg.Error as err:
        raise DocumentRegistryError(f"Database error while deleting: {str(err).strip()}", 503) from err

    files = _local_files(upload_dir, document_id)
    for path in files:
        path.unlink(missing_ok=True)
    # Chart images saved while parsing (structured and visual documents)
    figures = Path(upload_dir) / "figures" / document_id
    if figures.is_dir():
        shutil.rmtree(figures, ignore_errors=True)
        files.append(figures)

    if not deleted["document_deleted"] and not deleted["chunks_deleted"] and not files:
        raise DocumentRegistryError("Document not found.", 404)
    return {
        "document_id": document_id,
        "status": "deleted",
        "chunks_deleted": deleted["chunks_deleted"],
        "files_deleted": [path.name for path in files],
    }
