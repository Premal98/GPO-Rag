"""Tests for listing and deleting documents, with Supabase replaced by a fake."""

import io

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import config, document_registry, main
from app.document_registry import DocumentRegistryError, delete_document, list_documents
from app.pipeline import process_pdf_upload
from fakes import FakeChunkStore
from pdf_helpers import make_pdf, page_text

DIM = config.EMBEDDING_DIMENSION


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def store():
    store = FakeChunkStore(DIM)
    store.register("aaaaaaaaaaaaaaaa", "GPO-Annual-report.pdf", "narrative")
    store.register("bbbbbbbbbbbbbbbb", "Apple-iMac.pdf", "visual")
    return store


def test_lists_every_registered_document(store):
    result = list_documents(store=store)

    assert result["count"] == 2
    assert [(d["document_id"], d["filename"], d["profile"]) for d in result["documents"]] == [
        ("aaaaaaaaaaaaaaaa", "GPO-Annual-report.pdf", "narrative"),
        ("bbbbbbbbbbbbbbbb", "Apple-iMac.pdf", "visual"),
    ]


def test_empty_registry():
    assert list_documents(store=FakeChunkStore(DIM)) == {"documents": [], "count": 0}


@pytest.mark.parametrize("fake, message", [
    (FakeChunkStore(dimension=None), "001_create_document_chunks.sql"),
    (FakeChunkStore(DIM, migrated=False), "002_documents_and_chunk_metadata.sql"),
    (FakeChunkStore(DIM, error=psycopg.OperationalError("connection refused")), "Database error"),
])
def test_setup_problems_are_reported(fake, message):
    with pytest.raises(DocumentRegistryError) as err:
        list_documents(store=fake)
    assert err.value.status_code == 503
    assert message in err.value.message


def test_missing_database_url(monkeypatch):
    monkeypatch.setattr(config, "SUPABASE_DATABASE_URL", "")

    with pytest.raises(DocumentRegistryError) as err:
        list_documents()
    assert err.value.status_code == 503


def test_delete_removes_rows_and_uploaded_files(upload_dir):
    processed = process_pdf_upload(io.BytesIO(make_pdf([page_text(1)])), "report.pdf",
                                   "application/pdf", upload_dir, 10_000_000, 1000, 150)
    document_id = processed["document_id"]
    store = FakeChunkStore(DIM)
    store.rows["c1"] = {"chunk_id": "c1", "document_id": document_id}
    store.register(document_id)

    result = delete_document(document_id, upload_dir, store=store)

    assert result["status"] == "deleted"
    assert result["chunks_deleted"] == 1
    assert sorted(result["files_deleted"]) == [f"{document_id}.json", f"{document_id}_report.pdf"]
    assert store.documents == {} and store.rows == {}
    assert list(upload_dir.glob("*.pdf")) == []


def test_delete_unknown_document_is_404(upload_dir, store):
    with pytest.raises(DocumentRegistryError) as err:
        delete_document("0000000000000000", upload_dir, store=store)
    assert err.value.status_code == 404
    assert len(store.documents) == 2


def test_delete_rejects_ids_that_look_like_paths(upload_dir, store):
    with pytest.raises(DocumentRegistryError) as err:
        delete_document("../secret", upload_dir, store=store)
    assert err.value.status_code == 404


# ---------- The endpoints ----------

@pytest.fixture
def client(store, upload_dir, monkeypatch):
    monkeypatch.setattr(document_registry, "ChunkStore", lambda: store)
    monkeypatch.setattr(document_registry, "is_database_configured", lambda: True)
    return TestClient(main.app)


def test_get_documents_endpoint(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_delete_document_endpoint(client, store):
    response = client.delete("/documents/aaaaaaaaaaaaaaaa")
    missing = client.delete("/documents/aaaaaaaaaaaaaaaa")

    assert response.status_code == 200
    assert "aaaaaaaaaaaaaaaa" not in store.documents
    assert missing.status_code == 404


def test_documents_endpoint_reports_setup_problems(monkeypatch):
    monkeypatch.setattr(document_registry, "ChunkStore", lambda: FakeChunkStore(DIM, migrated=False))
    monkeypatch.setattr(document_registry, "is_database_configured", lambda: True)

    response = TestClient(main.app).get("/documents")

    assert response.status_code == 503
    assert "002" in response.json()["detail"]
