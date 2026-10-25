"""Tests for the ingestion pipeline, with Gemini and Supabase replaced by fakes."""

import io
import math
from unittest.mock import MagicMock

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import config, ingestion, main
from app import pipeline as pipeline_module
from app.document_store import load_result
from app.ingestion import IngestionError, ingest_document
from app.pipeline import process_pdf_upload
from app.vector_store import ChunkStore, to_vector_literal
from fakes import FakeChunkStore, FakeEmbedder
from pdf_helpers import long_text, make_pdf, page_text

DIM = config.EMBEDDING_DIMENSION


def no_sleep(seconds):
    pass


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def document_id(upload_dir):
    """Upload and chunk a 3-page PDF (page 3 is blank), returning its document ID."""
    pdf = make_pdf([long_text(400), page_text(2), None])
    result = process_pdf_upload(io.BytesIO(pdf), "Annual-Report.pdf", "application/pdf",
                                upload_dir, 10_000_000, chunk_size=1000, chunk_overlap=150)
    return result["document_id"]


def run(document_id, upload_dir, store, embedder=None, force=False):
    return ingest_document(document_id, upload_dir, force=force,
                           embedder=embedder or FakeEmbedder(DIM), store=store, sleep=no_sleep)


# ---------- Successful ingestion ----------

def test_chunks_are_stored_with_embeddings_and_metadata(document_id, upload_dir):
    store = FakeChunkStore(DIM)

    result = run(document_id, upload_dir, store)

    assert result["status"] == "ingested"
    assert result["chunks_inserted"] == result["chunk_count"] == len(store.rows)
    for row in store.rows.values():
        assert row["document_id"] == document_id
        assert row["source_filename"] == "Annual-Report.pdf"
        assert row["page_number"] in (1, 2)
        assert row["chunk_id"] == row["metadata"]["chunk_id"]
        assert row["metadata"]["page"] == row["page_number"]
        assert row["metadata"]["embedding_model"] == config.EMBEDDING_MODEL
        assert len(row["embedding"]) == DIM
        assert math.isclose(math.sqrt(sum(x * x for x in row["embedding"])), 1.0)
        # The stored content is the raw chunk, without the embedding prefix
        assert not row["content"].startswith("title:")
        # Plain page chunks get the default metadata columns
        assert row["chunk_type"] == "text"
        assert row["page_end"] == row["page_number"]
        assert row["section_path"] is None and row["parent_chunk_id"] is None


def test_ingesting_registers_the_document(document_id, upload_dir):
    store = FakeChunkStore(DIM)

    result = run(document_id, upload_dir, store)

    registered = store.documents[document_id]
    assert registered["filename"] == "Annual-Report.pdf"
    assert registered["profile"] == result["profile"] == load_result(upload_dir, document_id)["profile"]
    assert registered["page_count"] == 3
    assert registered["status"] == "ready"


def test_profile_chosen_at_upload_is_registered(upload_dir):
    processed = process_pdf_upload(io.BytesIO(make_pdf([page_text(1)])), "env.pdf", "application/pdf",
                                   upload_dir, 10_000_000, chunk_size=1000, chunk_overlap=150,
                                   profile="visual")
    store = FakeChunkStore(DIM)

    run(processed["document_id"], upload_dir, store)

    assert store.documents[processed["document_id"]]["profile"] == "visual"


def test_embedding_text_uses_title_prefix_for_gemini_embedding_2(document_id, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-2")
    embedder = FakeEmbedder(DIM)

    run(document_id, upload_dir, FakeChunkStore(DIM), embedder=embedder)

    sent = [text for call in embedder.calls for text in call]
    assert all(text.startswith("title: Annual-Report | text: ") for text in sent)


def test_chunks_are_embedded_in_batches(document_id, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)
    embedder = FakeEmbedder(DIM)

    result = run(document_id, upload_dir, FakeChunkStore(DIM), embedder=embedder)

    assert all(len(call) <= 2 for call in embedder.calls)
    assert sum(len(call) for call in embedder.calls) == result["chunk_count"]


# ---------- Duplicate protection ----------

def test_second_ingest_does_nothing_and_skips_the_api(document_id, upload_dir):
    store = FakeChunkStore(DIM)
    run(document_id, upload_dir, store)
    embedder = FakeEmbedder(DIM)

    result = run(document_id, upload_dir, store, embedder=embedder)

    assert result["status"] == "already_ingested"
    assert result["chunks_inserted"] == 0
    assert embedder.calls == []  # no API quota used


def test_force_replaces_existing_rows(document_id, upload_dir):
    store = FakeChunkStore(DIM)
    first = run(document_id, upload_dir, store)

    result = run(document_id, upload_dir, store, force=True)

    assert result["status"] == "reingested"
    assert len(store.rows) == first["chunk_count"]


def test_document_ingested_by_another_request_meanwhile(document_id, upload_dir):
    store = FakeChunkStore(DIM)
    store.count_chunks = lambda doc_id: 0  # looks empty at the start...
    store.rows["other"] = {"document_id": document_id, "chunk_id": "other"}  # ...but isn't at save time

    result = run(document_id, upload_dir, store)

    assert result["status"] == "already_ingested"


# ---------- Errors ----------

def test_unknown_document(upload_dir):
    with pytest.raises(IngestionError) as err:
        run("0000000000000000", upload_dir, FakeChunkStore(DIM))
    assert err.value.status_code == 404


def test_scanned_document_has_nothing_to_ingest(upload_dir):
    result = process_pdf_upload(io.BytesIO(make_pdf([None, None])), "scan.pdf", "application/pdf",
                                upload_dir, 10_000_000, chunk_size=1000, chunk_overlap=150)

    with pytest.raises(IngestionError) as err:
        run(result["document_id"], upload_dir, FakeChunkStore(DIM))
    assert err.value.status_code == 422
    assert "OCR" in err.value.message


def test_dimension_mismatch_stops_before_calling_the_api(document_id, upload_dir):
    embedder = FakeEmbedder(DIM)

    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, FakeChunkStore(dimension=768), embedder=embedder)

    assert err.value.status_code == 409
    assert embedder.calls == []


def test_missing_table(document_id, upload_dir):
    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, FakeChunkStore(dimension=None))
    assert err.value.status_code == 503
    assert "001_create_document_chunks.sql" in err.value.message


def test_database_without_sql_002_stops_before_calling_the_api(document_id, upload_dir):
    embedder = FakeEmbedder(DIM)

    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, FakeChunkStore(DIM, migrated=False), embedder=embedder)

    assert err.value.status_code == 503
    assert "002_documents_and_chunk_metadata.sql" in err.value.message
    assert embedder.calls == []


def test_database_outage(document_id, upload_dir):
    store = FakeChunkStore(DIM, error=psycopg.OperationalError("connection refused"))

    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, store)
    assert err.value.status_code == 503


def test_embedding_failure_saves_nothing(document_id, upload_dir):
    store = FakeChunkStore(DIM)
    embedder = FakeEmbedder(DIM, failures=[ValueError("API key not valid")])

    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, store, embedder=embedder)

    assert err.value.status_code == 502
    assert store.rows == {}


def test_quota_error_returns_429_and_saves_nothing(document_id, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 1)
    store = FakeChunkStore(DIM)
    quota = RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 5s.")
    embedder = FakeEmbedder(DIM, failures=[quota, quota])

    with pytest.raises(IngestionError) as err:
        run(document_id, upload_dir, store, embedder=embedder)

    assert err.value.status_code == 429
    assert "quota" in err.value.message
    assert store.rows == {}


def test_placeholder_credentials_are_reported_not_used(document_id, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    with pytest.raises(IngestionError) as err:
        ingest_document(document_id, upload_dir)  # real embedder and store
    assert err.value.status_code == 503
    assert "GOOGLE_API_KEY" in err.value.message


# ---------- The endpoint ----------

def test_ingest_endpoint(document_id, upload_dir, monkeypatch):
    store = FakeChunkStore(DIM)
    real_ingest = ingestion.ingest_document

    def fake_ingest(doc_id, folder, force=False):
        return real_ingest(doc_id, folder, force=force, embedder=FakeEmbedder(DIM),
                           store=store, sleep=no_sleep)
    monkeypatch.setattr(main, "ingest_document", fake_ingest)
    client = TestClient(main.app)

    first = client.post(f"/documents/{document_id}/ingest")
    second = client.post(f"/documents/{document_id}/ingest")
    forced = client.post(f"/documents/{document_id}/ingest?force=true")
    missing = client.post("/documents/0000000000000000/ingest")

    assert first.status_code == 200 and first.json()["status"] == "ingested"
    assert second.json()["status"] == "already_ingested"
    assert forced.json()["status"] == "reingested"
    assert missing.status_code == 404


# ---------- The real ChunkStore, with a mocked database connection ----------

def test_vector_literal_format():
    assert to_vector_literal([0.5, -1, 2.25]) == "[0.5,-1.0,2.25]"


def make_mock_connection(existing_count):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = (existing_count,)
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.rowcount = 2
    return conn, cursor


DOCUMENT = {"document_id": "doc", "filename": "a.pdf", "profile": "narrative", "page_count": 1}


def sample_rows():
    return [
        {"chunk_id": f"doc_p1_c{i}", "document_id": "doc", "content": f"text {i}",
         "source_filename": "a.pdf", "page_number": 1, "chunk_index": i,
         "metadata": {"page": 1}, "embedding": [0.6, 0.8], "chunk_type": "text",
         "section_path": None, "page_end": 1, "parent_chunk_id": None, "printed_page": None}
        for i in range(2)
    ]


def test_save_chunks_inserts_in_one_transaction_with_lock():
    conn, cursor = make_mock_connection(existing_count=0)

    result = ChunkStore(connect=lambda: conn).save_chunks(DOCUMENT, sample_rows())

    assert result == {"skipped": False, "inserted": 2, "existing": 0}
    conn.transaction.assert_called_once()
    statements = [(str(c.args[0]), c.args[1] if len(c.args) > 1 else None)
                  for c in conn.execute.call_args_list]
    sql_run = " ".join(sql for sql, _ in statements)
    assert "pg_advisory_xact_lock" in sql_run
    assert "delete" not in sql_run
    # The document is registered before its chunks, and its chunk count set after
    upsert = next(i for i, (sql, _) in enumerate(statements) if "insert into public.documents" in sql)
    assert statements[upsert][1] == DOCUMENT
    assert "set chunk_count" in statements[-1][0]
    assert statements[-1][1] == (2, "doc")
    insert_sql, params = cursor.executemany.call_args.args
    assert "on conflict (chunk_id) do nothing" in insert_sql
    assert params[0][0] == "doc_p1_c0"
    assert params[0][7] == "[0.6,0.8]"
    assert params[0][8:] == ("text", None, 1, None, None)


def test_save_chunks_skips_existing_document():
    conn, cursor = make_mock_connection(existing_count=5)

    result = ChunkStore(connect=lambda: conn).save_chunks(DOCUMENT, sample_rows())

    assert result["skipped"] is True
    cursor.executemany.assert_not_called()
    sql_run = " ".join(str(c.args[0]) for c in conn.execute.call_args_list)
    assert "public.documents" not in sql_run


def test_save_chunks_replace_deletes_first():
    conn, cursor = make_mock_connection(existing_count=5)

    result = ChunkStore(connect=lambda: conn).save_chunks(DOCUMENT, sample_rows(), replace=True)

    assert result["skipped"] is False
    sql_run = " ".join(str(c.args[0]) for c in conn.execute.call_args_list)
    assert "delete from public.document_chunks" in sql_run
    cursor.executemany.assert_called_once()


# ---------- Structure-aware chunks (Docling is faked, see conftest.py) ----------

def test_structured_parents_are_stored_without_embeddings(upload_dir):
    pdf = make_pdf([long_text(500, prefix="alpha"), long_text(500, prefix="beta")])
    processed = process_pdf_upload(io.BytesIO(pdf), "wb.pdf", "application/pdf", upload_dir,
                                   10_000_000, 1000, 150, profile="structured")
    store, embedder = FakeChunkStore(DIM), FakeEmbedder(DIM)

    run(processed["document_id"], upload_dir, store, embedder=embedder)

    parents = [r for r in store.rows.values() if r["embedding"] is None]
    children = [r for r in store.rows.values() if r["embedding"] is not None]
    assert parents and children
    assert all(r["parent_chunk_id"] in {p["chunk_id"] for p in parents} for r in children)
    assert sum(len(call) for call in embedder.calls) == len(children)  # parents are not embedded
    assert store.documents[processed["document_id"]]["profile"] == "structured"


def test_section_path_is_part_of_the_embedding_title(upload_dir, monkeypatch, layout_parser):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-2")
    original = layout_parser.__call__

    def with_heading(*args, **kwargs):
        result = original(*args, **kwargs)
        result["elements"].insert(0, {"type": "heading", "text": "2 Outlook", "page": 1,
                                      "level": 1, "bbox": None, "caption": None})
        return result
    monkeypatch.setattr(pipeline_module, "parse_pdf", with_heading)
    processed = process_pdf_upload(io.BytesIO(make_pdf([page_text(1)])), "WB.pdf",
                                   "application/pdf", upload_dir, 10_000_000, 1000, 150,
                                   profile="visual")
    embedder = FakeEmbedder(DIM)

    run(processed["document_id"], upload_dir, FakeChunkStore(DIM), embedder=embedder)

    assert embedder.calls[0][0].startswith("title: WB > 2 Outlook | text: ")
