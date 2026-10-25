"""Tests for background processing (POST /documents), with Gemini, Supabase and Docling faked."""

import io

import pytest
from fastapi.testclient import TestClient

from app import config, ingestion, jobs, main
from app.document_registry import DocumentRegistryError, get_document
from app.jobs import recover_interrupted, run_document_job, submit_document
from app.retrieval import RetrievalError, search_chunks
from fakes import FakeChunkStore, FakeEmbedder
from pdf_helpers import make_pdf

DIM = config.EMBEDDING_DIMENSION
TEXT = "The office served many libraries and readers across the country this year."


def no_sleep(seconds):
    pass


def fake_ingest(document_id, upload_dir, force=False, store=None, processed=None):
    return ingestion.ingest_document(document_id, upload_dir, force=force, store=store,
                                     embedder=FakeEmbedder(DIM), sleep=no_sleep, processed=processed)


@pytest.fixture(autouse=True)
def no_active_jobs():
    """Jobs a test queued but never ran must not leak into the next test."""
    jobs._active.clear()
    yield
    jobs._active.clear()


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")  # nothing calls Gemini
    return tmp_path


@pytest.fixture
def store():
    return FakeChunkStore(DIM)


def run_now(job):
    """Run the job synchronously instead of on the background worker."""
    run_document_job(*job, ingest=fake_ingest)


def submit(store, upload_dir, pdf=None, profile=None, force=False, run=run_now):
    return submit_document(io.BytesIO(pdf or make_pdf([TEXT])), "report.pdf", "application/pdf",
                           profile=profile, force=force, store=store, upload_dir=upload_dir,
                           run=run)


# ---------- The whole flow ----------

def test_document_goes_from_uploaded_to_ready(store, upload_dir):
    result = submit(store, upload_dir)

    document_id = result["document_id"]
    assert result["accepted"] is True
    assert result["status_url"] == f"/documents/{document_id}"
    assert [s for _, s in store.status_history] == ["uploaded", "embedding"]
    assert store.documents[document_id]["status"] == "ready"
    assert store.count_chunks(document_id) > 0


def test_layout_profiles_report_the_parsing_stage(store, upload_dir):
    submit(store, upload_dir, profile="visual")

    assert [s for _, s in store.status_history] == ["uploaded", "parsing", "embedding"]
    assert next(iter(store.documents.values()))["profile"] == "visual"


def test_suggested_profile_is_recorded(store, upload_dir):
    result = submit(store, upload_dir, pdf=make_pdf(["Year 2019 2020 2021 CO2 25.1 23.4 20.0"]))

    assert store.documents[result["document_id"]]["profile"] == "visual"


def test_ready_document_is_not_processed_again(store, upload_dir):
    first = submit(store, upload_dir)
    queued = []

    again = submit(store, upload_dir, run=queued.append)
    forced = submit(store, upload_dir, force=True, run=queued.append)

    assert again["accepted"] is False and again["status"] == "ready"
    assert forced["accepted"] is True
    assert len(queued) == 1 and queued[0][0] == first["document_id"]


def test_a_new_profile_processes_a_ready_document_again(store, upload_dir):
    submit(store, upload_dir)
    queued = []

    result = submit(store, upload_dir, profile="structured", run=queued.append)

    assert result["accepted"] is True and len(queued) == 1


def test_document_being_processed_is_not_queued_twice(store, upload_dir):
    queued = []
    submit(store, upload_dir, run=queued.append)  # stays "uploaded": the job never runs

    second = submit(store, upload_dir, run=queued.append)

    assert second["accepted"] is False
    assert second["status"] == "processing"
    assert len(queued) == 1


# ---------- Failures ----------

def test_scanned_pdf_ends_as_failed(store, upload_dir):
    result = submit(store, upload_dir, pdf=make_pdf([None, None]))

    document = store.documents[result["document_id"]]
    assert document["status"] == "failed"
    assert "OCR" in document["status_message"]


def test_errors_during_the_job_are_recorded(store, upload_dir):
    def broken_ingest(*args, **kwargs):
        raise RuntimeError("disk full")

    result = submit(store, upload_dir,
                    run=lambda job: run_document_job(*job, ingest=broken_ingest))

    document = store.documents[result["document_id"]]
    assert document["status"] == "failed"
    assert "disk full" in document["status_message"]


def test_corrupted_pdf_is_rejected_before_queueing(store, upload_dir):
    from app.pdf_ingestion import PDFIngestionError
    queued = []

    with pytest.raises(PDFIngestionError):
        submit(store, upload_dir, pdf=b"%PDF-1.4\nnot a real pdf", run=queued.append)

    assert queued == [] and store.documents == {}
    assert list(upload_dir.glob("*.pdf")) == []


def test_missing_settings_are_reported_before_saving(store, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    with pytest.raises(DocumentRegistryError) as err:
        submit(store, upload_dir)
    assert err.value.status_code == 503
    assert list(upload_dir.glob("*.pdf")) == []

    with pytest.raises(DocumentRegistryError) as err:
        monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")
        submit(FakeChunkStore(DIM, migrated=False), upload_dir)
    assert "002" in err.value.message


def test_restart_marks_unfinished_documents_as_failed(store):
    store.register("aaaaaaaaaaaaaaaa")
    store.set_document_status("bbbbbbbbbbbbbbbb", "b.pdf", "parsing")

    assert recover_interrupted(store) == 1
    assert store.documents["bbbbbbbbbbbbbbbb"]["status"] == "failed"
    assert store.documents["aaaaaaaaaaaaaaaa"]["status"] == "ready"


def test_recovery_without_a_database_does_nothing():
    assert recover_interrupted(FakeChunkStore(DIM, migrated=False)) == 0


# ---------- Status and chat while processing ----------

def test_document_status_includes_processing_details(store, upload_dir):
    document_id = submit(store, upload_dir, profile="visual")["document_id"]

    status = get_document(document_id, upload_dir, store=store)

    assert status["status"] == "ready" and status["chat_ready"] is True
    assert status["processing"]["parser"] == "docling"
    assert status["processing"]["chunk_count"] > 0


def test_uploaded_but_not_ingested_document(store, upload_dir):
    from app.pipeline import process_pdf_upload
    processed = process_pdf_upload(io.BytesIO(make_pdf([TEXT])), "r.pdf", "application/pdf",
                                   upload_dir, 10_000_000, 1000, 150)

    status = get_document(processed["document_id"], upload_dir, store=store)

    assert status["status"] == "not_ingested" and status["chat_ready"] is False


def test_unknown_document_status_is_404(store, upload_dir):
    with pytest.raises(DocumentRegistryError) as err:
        get_document("0000000000000000", upload_dir, store=store)
    assert err.value.status_code == 404


@pytest.mark.parametrize("status, message", [("parsing", "still being processed"),
                                             ("failed", "failed")])
def test_chat_waits_for_the_document_to_be_ready(store, status, message):
    store.set_document_status("aaaaaaaaaaaaaaaa", "a.pdf", status, "Docling crashed.")
    embedder = FakeEmbedder(DIM)

    with pytest.raises(RetrievalError) as err:
        search_chunks("question", document_id="aaaaaaaaaaaaaaaa", embedder=embedder, store=store)

    assert err.value.status_code == 409
    assert message in err.value.message
    assert embedder.calls == []


# ---------- The endpoints ----------

@pytest.fixture
def client(store, upload_dir, monkeypatch):
    from app import document_registry
    monkeypatch.setattr(document_registry, "ChunkStore", lambda: store)
    monkeypatch.setattr(document_registry, "is_database_configured", lambda: True)
    monkeypatch.setattr(jobs, "_executor", type("Inline", (), {
        "submit": staticmethod(lambda fn, *args: fn(*args, ingest=fake_ingest))})())
    return TestClient(main.app)


def test_post_documents_returns_202_then_status_shows_ready(client):
    response = client.post("/documents", files={"file": ("report.pdf", make_pdf([TEXT]),
                                                         "application/pdf")})

    assert response.status_code == 202
    body = response.json()
    status = client.get(body["status_url"]).json()
    assert status["status"] == "ready"
    assert status["filename"] == "report.pdf"


def test_post_documents_for_a_ready_document_returns_200(client):
    files = {"file": ("report.pdf", make_pdf([TEXT]), "application/pdf")}
    client.post("/documents", files=files)

    again = client.post("/documents", files=files)

    assert again.status_code == 200
    assert again.json()["accepted"] is False


def test_post_documents_rejects_invalid_files_and_profiles(client):
    bad_file = client.post("/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    bad_profile = client.post("/documents?profile=fancy",
                              files={"file": ("x.pdf", make_pdf([TEXT]), "application/pdf")})

    assert bad_file.status_code == 400
    assert bad_profile.status_code == 422


def test_get_unknown_document_is_404(client):
    assert client.get("/documents/0000000000000000").status_code == 404


# ---------- Robustness (stuck statuses, races) ----------

def test_a_stale_busy_status_does_not_block_a_new_upload(store, upload_dir):
    # E.g. recording "failed" itself failed: the row says parsing, but no job runs
    document_id = submit(store, upload_dir, run=lambda job: None)["document_id"]
    jobs._active.clear()
    store.documents[document_id]["status"] = "parsing"

    result = submit(store, upload_dir)

    assert result["accepted"] is True
    assert store.documents[document_id]["status"] == "ready"


def test_job_finishing_releases_the_document(store, upload_dir):
    document_id = submit(store, upload_dir)["document_id"]

    assert not jobs.is_active(document_id)


def test_failure_that_cannot_be_recorded_is_logged_not_raised(store, upload_dir, monkeypatch, caplog):
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    def broken_ingest(*args, **kwargs):
        raise RuntimeError("disk full")

    def run(job):
        store.error = None
        original = store.set_document_status

        def failing_status(document_id, filename, status, *args, **kwargs):
            if status == "failed":
                raise RuntimeError("database gone")
            return original(document_id, filename, status, *args, **kwargs)
        store.set_document_status = failing_status
        run_document_job(*job, ingest=broken_ingest)  # must not raise

    document_id = submit(store, upload_dir, run=run)["document_id"]

    assert "Could not mark document" in caplog.text
    assert not jobs.is_active(document_id)


def test_job_ingests_its_own_result_not_the_file_on_disk(store, upload_dir):
    seen = []

    def recording_ingest(document_id, upload_dir, force=False, store=None, processed=None):
        seen.append(processed["profile"])
        return fake_ingest(document_id, upload_dir, force, store, processed)

    submit(store, upload_dir, profile="visual",
           run=lambda job: run_document_job(*job, ingest=recording_ingest))

    assert seen == ["visual"]


def test_step_by_step_endpoints_refuse_a_document_being_processed(client, store, upload_dir):
    from app.pipeline import process_pdf_upload
    pdf = make_pdf([TEXT])
    document_id = process_pdf_upload(io.BytesIO(pdf), "r.pdf", "application/pdf", upload_dir,
                                     10_000_000, 1000, 150)["document_id"]
    jobs._active.add(document_id)

    upload = client.post("/documents/upload", files={"file": ("r.pdf", pdf, "application/pdf")})
    ingest = client.post(f"/documents/{document_id}/ingest")
    delete = client.delete(f"/documents/{document_id}")

    assert upload.status_code == ingest.status_code == delete.status_code == 409
    assert "being processed" in delete.json()["detail"]


def test_database_errors_while_submitting_are_503(upload_dir):
    import psycopg
    store = FakeChunkStore(DIM)
    store.set_document_status = lambda *a, **k: (_ for _ in ()).throw(psycopg.OperationalError("gone"))

    with pytest.raises(DocumentRegistryError) as err:
        submit(store, upload_dir)
    assert err.value.status_code == 503
