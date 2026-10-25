"""Real end-to-end tests: Gemini API + Supabase. Skipped by default.

They upload a tiny generated PDF, embed its chunks with the real Gemini API,
store them in your real document_chunks table, search them, ask the real chat
model about them, then delete the rows they created. They use a few API
calls of quota.

To run them:
  1. Set GOOGLE_API_KEY and SUPABASE_DATABASE_URL in .env
  2. Run the SQL setup script in Supabase
  3. In PowerShell:
       $env:RUN_INTEGRATION_TESTS = "1"; python -m pytest -m integration -v -s
       Remove-Item Env:RUN_INTEGRATION_TESTS
"""

import io
import os
import uuid

import pytest

from app import config
from app.chat import answer_question
from app.database import get_connection, is_database_configured
from app.embeddings import is_embedding_configured
from app.ingestion import ingest_document
from app.pipeline import process_pdf_upload
from app.retrieval import search_chunks
from app.vector_store import ChunkStore
from pdf_helpers import long_text, make_pdf

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1"
        or not is_embedding_configured()
        or not is_database_configured(),
        reason="Set RUN_INTEGRATION_TESTS=1 and real credentials in .env to run",
    ),
]

# Unique text, so these tests never match one of your real documents
MARKER = uuid.uuid4().hex[:8]
FACT_PAGE_TEXT = f"The internal project code for test {MARKER} is BLUE HERON 42."


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """Upload and ingest a 2-page PDF once; delete its rows afterwards."""
    folder = tmp_path_factory.mktemp("uploads")
    pdf = make_pdf([f"Integration test {MARKER}. " + long_text(200), FACT_PAGE_TEXT])
    processed = process_pdf_upload(io.BytesIO(pdf), f"integration-{MARKER}.pdf", "application/pdf",
                                   folder, 10_000_000, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    document_id = processed["document_id"]
    try:
        first = ingest_document(document_id, folder)
        print("\nFirst ingest:", first)
        yield {"folder": folder, "processed": processed, "document_id": document_id,
               "first_ingest": first}
    finally:
        deleted = ChunkStore().delete_document(document_id)
        print(f"Cleaned up {deleted['chunks_deleted']} test rows.")


def test_real_ingestion_round_trip(ingested):
    processed, document_id = ingested["processed"], ingested["document_id"]
    assert ingested["first_ingest"]["status"] == "ingested"
    assert ingested["first_ingest"]["chunks_inserted"] == processed["chunk_count"]

    second = ingest_document(document_id, ingested["folder"])
    print("Second ingest:", second)
    assert second["status"] == "already_ingested"

    with get_connection() as conn:
        rows = conn.execute(
            """
            select chunk_id, source_filename, page_number, vector_dims(embedding), metadata
            from public.document_chunks where document_id = %s order by chunk_index
            """,
            (document_id,),
        ).fetchall()
    assert len(rows) == processed["chunk_count"]
    for chunk_id, filename, page, dims, metadata in rows:
        assert filename == f"integration-{MARKER}.pdf"
        assert page in (1, 2)
        assert dims == config.EMBEDDING_DIMENSION
        assert metadata["chunk_id"] == chunk_id
    print(f"Verified {len(rows)} rows with {config.EMBEDDING_DIMENSION}-dim embeddings.")


def test_real_retrieval_finds_the_right_page(ingested):
    result = search_chunks(f"What is the internal project code for test {MARKER}?",
                           top_k=3, document_id=ingested["document_id"])
    print("Search:", [(r["page_number"], r["similarity"]) for r in result["results"]])

    assert result["status"] == "ok"
    assert result["results"][0]["page_number"] == 2
    assert result["results"][0]["filename"] == f"integration-{MARKER}.pdf"
    assert result["warnings"] == []


def test_real_chat_answers_with_a_citation(ingested):
    result = answer_question(f"What is the internal project code for test {MARKER}?",
                             document_id=ingested["document_id"])
    print("Chat:", result["status"], result["answer"], result["citations"])

    assert result["status"] == "answered"
    assert "BLUE HERON 42" in result["answer"].upper()
    assert any(s["page_number"] == 2 for s in result["sources"])


def test_real_chat_says_when_the_answer_is_not_in_the_document(ingested):
    result = answer_question("What was the population of Paris in 1850?",
                             document_id=ingested["document_id"])
    print("Chat:", result["status"], result["answer"])

    assert result["status"] in ("not_supported", "no_context")
    assert result["supported"] is False
