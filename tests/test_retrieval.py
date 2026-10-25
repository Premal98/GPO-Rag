"""Tests for semantic retrieval, with Gemini and Supabase replaced by fakes."""

import math
from unittest.mock import MagicMock

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient

from app import config, main
from app.retrieval import RetrievalError, search_chunks
from app.vector_store import ChunkStore
from fakes import FakeChunkStore, FakeEmbedder

# Small vectors keep the expected rankings easy to read
DIM = 3


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error from API")
        self.code = code


def no_sleep(seconds):
    pass


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", DIM)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 2)
    monkeypatch.setattr(config, "RETRIEVAL_MAX_TOP_K", 10)
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SIMILARITY", None)


def add_chunk(store, chunk_id, embedding, document_id="doc1", filename="Annual-Report.pdf",
              page=1, model="gemini-embedding-2"):
    store.rows[chunk_id] = {
        "chunk_id": chunk_id, "document_id": document_id, "content": f"Text of {chunk_id}",
        "source_filename": filename, "page_number": page, "chunk_index": len(store.rows),
        "metadata": {"page": page, "embedding_model": model}, "embedding": embedding,
    }
    if document_id not in store.documents:
        store.register(document_id, filename)


@pytest.fixture
def store():
    """Three chunks pointing in different directions; the query points along x."""
    store = FakeChunkStore(DIM)
    add_chunk(store, "revenue", [1.0, 0.1, 0.0], page=6)       # nearly the same direction
    add_chunk(store, "staff", [1.0, 1.0, 0.0], page=5)          # 45 degrees away
    add_chunk(store, "weather", [0.0, 0.0, 1.0], page=9)        # unrelated
    return store


def search(store, query="What was the revenue?", embedder=None, **kwargs):
    embedder = embedder or FakeEmbedder(DIM, query_vector=[2.0, 0.0, 0.0])
    return search_chunks(query, embedder=embedder, store=store, sleep=no_sleep, **kwargs)


# ---------- Successful searches ----------

def test_results_are_ranked_by_cosine_similarity(store):
    result = search(store, top_k=3)

    assert result["status"] == "ok"
    assert result["metric"] == "cosine"
    assert [r["chunk_id"] for r in result["results"]] == ["revenue", "staff", "weather"]
    assert [r["rank"] for r in result["results"]] == [1, 2, 3]
    staff = result["results"][1]
    assert math.isclose(staff["similarity"], 1 / math.sqrt(2), abs_tol=1e-6)
    assert math.isclose(staff["similarity"] + staff["distance"], 1.0, abs_tol=1e-6)
    assert math.isclose(result["results"][2]["similarity"], 0.0, abs_tol=1e-6)


def test_results_include_content_filename_and_page(store):
    top = search(store)["results"][0]

    assert top["content"] == "Text of revenue"
    assert top["filename"] == "Annual-Report.pdf"
    assert top["page_number"] == 6
    assert top["page_end"] == 6  # defaults to page_number for single-page chunks
    assert top["document_id"] == "doc1"
    assert top["chunk_type"] == "text"
    assert top["section_path"] is None


def test_top_k_defaults_to_config_and_can_be_overridden(store):
    default = search(store)
    one = search(store, top_k=1)

    assert default["top_k"] == 2 and default["result_count"] == 2
    assert one["result_count"] == 1
    # More candidates than top_k are fetched, to be merged and reranked
    assert [s[1] for s in store.searches] == [config.RETRIEVAL_CANDIDATES] * 2


def test_query_uses_the_gemini_embedding_2_query_prefix(store):
    embedder = FakeEmbedder(DIM, query_vector=[1.0, 0.0, 0.0])

    search(store, query="  What was the revenue?  ", embedder=embedder)

    assert embedder.calls == [["task: search result | query: What was the revenue?"]]


def test_older_models_get_the_plain_query(store, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-001")
    for row in store.rows.values():
        row["metadata"]["embedding_model"] = "gemini-embedding-001"
    embedder = FakeEmbedder(DIM, query_vector=[1.0, 0.0, 0.0])

    result = search(store, query="revenue", embedder=embedder)

    assert embedder.calls == [["revenue"]]
    assert result["warnings"] == []


def test_query_vector_is_normalized_like_stored_chunks(store):
    search(store)  # the fake model returns [2, 0, 0]

    query_vector = store.searches[0][0]
    assert query_vector == [1.0, 0.0, 0.0]


def test_search_can_be_limited_to_one_document(store):
    add_chunk(store, "other-doc", [1.0, 0.0, 0.0], document_id="doc2", filename="Other.pdf")

    everywhere = search(store, query="q", top_k=4)
    only_doc1 = search(store, query="q", top_k=4, document_id="doc1")

    assert "other-doc" in [r["chunk_id"] for r in everywhere["results"]]
    assert {r["document_id"] for r in only_doc1["results"]} == {"doc1"}
    assert store.searches[1][2] == "doc1"


def test_min_similarity_hides_weak_matches(store):
    result = search(store, top_k=3, min_similarity=0.5)

    assert [r["chunk_id"] for r in result["results"]] == ["revenue", "staff"]


def test_min_similarity_default_comes_from_config(store, monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SIMILARITY", 0.9)

    result = search(store, top_k=3)

    assert result["min_similarity"] == 0.9
    assert [r["chunk_id"] for r in result["results"]] == ["revenue"]


def test_no_chunk_reaches_min_similarity(store):
    embedder = FakeEmbedder(DIM, query_vector=[0.0, -1.0, 0.0])

    result = search(store, embedder=embedder, min_similarity=0.5)

    assert result["status"] == "no_matches"
    assert result["results"] == []
    assert "lowering min_similarity" in result["message"]


def test_chunks_from_a_different_model_produce_a_warning(store):
    add_chunk(store, "old", [1.0, 0.0, 0.0], model="gemini-embedding-001")

    result = search(store)

    assert len(result["warnings"]) == 1
    assert "gemini-embedding-001" in result["warnings"][0]


# ---------- Empty database ----------

def test_empty_database_is_reported_clearly():
    result = search(FakeChunkStore(DIM))

    assert result["status"] == "empty_database"
    assert result["results"] == []
    assert "Upload a PDF and ingest it" in result["message"]


def test_unknown_document_returns_404_before_calling_the_api(store):
    embedder = FakeEmbedder(DIM)

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder, document_id="not-ingested")

    assert err.value.status_code == 404
    assert "GET /documents" in err.value.message
    assert "/documents/not-ingested/ingest" in err.value.message
    assert embedder.calls == [] and store.searches == []


def test_registered_document_without_embedded_chunks_points_to_the_ingest_endpoint(store):
    store.register("doc9")

    result = search(store, document_id="doc9")

    assert result["status"] == "empty_database"
    assert "/documents/doc9/ingest" in result["message"]


def test_rows_without_embeddings_are_not_searched(store):
    for row in store.rows.values():
        row["embedding"] = None

    assert search(store)["status"] == "empty_database"


# ---------- Invalid input ----------

@pytest.mark.parametrize("kwargs", [
    {"query": "   "},
    {"top_k": 0},
    {"top_k": 11},
    {"min_similarity": 1.5},
])
def test_invalid_input_is_rejected_before_calling_the_api(store, kwargs):
    embedder = FakeEmbedder(DIM)
    kwargs.setdefault("query", "revenue")

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder, **kwargs)

    assert err.value.status_code == 422
    assert embedder.calls == []


# ---------- Missing credentials and setup problems ----------

def test_missing_google_api_key(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    with pytest.raises(RetrievalError) as err:
        search_chunks("revenue")  # real embedder and store
    assert err.value.status_code == 503
    assert "GOOGLE_API_KEY" in err.value.message


def test_missing_database_url(monkeypatch):
    monkeypatch.setattr(config, "SUPABASE_DATABASE_URL", "")
    embedder = FakeEmbedder(DIM)

    with pytest.raises(RetrievalError) as err:
        search_chunks("revenue", embedder=embedder)
    assert err.value.status_code == 503
    assert "SUPABASE_DATABASE_URL" in err.value.message
    assert embedder.calls == []


def test_missing_table():
    with pytest.raises(RetrievalError) as err:
        search(FakeChunkStore(dimension=None))
    assert err.value.status_code == 503
    assert "001_create_document_chunks.sql" in err.value.message


def test_database_without_sql_002_is_reported_before_calling_the_api():
    embedder = FakeEmbedder(DIM)

    with pytest.raises(RetrievalError) as err:
        search(FakeChunkStore(DIM, migrated=False), embedder=embedder)

    assert err.value.status_code == 503
    assert "002_documents_and_chunk_metadata.sql" in err.value.message
    assert embedder.calls == []


def test_dimension_mismatch_stops_before_calling_the_api():
    embedder = FakeEmbedder(DIM)

    with pytest.raises(RetrievalError) as err:
        search(FakeChunkStore(dimension=768), embedder=embedder)

    assert err.value.status_code == 409
    assert embedder.calls == []


def test_database_outage():
    store = FakeChunkStore(DIM, error=psycopg.OperationalError("connection refused"))

    with pytest.raises(RetrievalError) as err:
        search(store)
    assert err.value.status_code == 503


def test_database_error_during_search(store):
    store.search = MagicMock(side_effect=psycopg.OperationalError("server closed the connection"))

    with pytest.raises(RetrievalError) as err:
        search(store)
    assert err.value.status_code == 503
    assert "while searching" in err.value.message


# ---------- Embedding problems ----------

def test_embedding_failure(store):
    embedder = FakeEmbedder(DIM, failures=[ValueError("API key not valid")])

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder)

    assert err.value.status_code == 502
    assert store.searches == []


def test_wrong_query_dimension_is_rejected(store):
    embedder = FakeEmbedder(DIM, query_vector=[1.0] * 768)

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder)
    assert err.value.status_code == 502
    assert "768-dimension" in err.value.message


def test_embedding_timeout_returns_504(store):
    embedder = FakeEmbedder(DIM, failures=[httpx.ReadTimeout("timed out")])

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder)
    assert err.value.status_code == 504
    assert "EMBEDDING_TIMEOUT_SECONDS" in err.value.message
    assert len(embedder.calls) == 1  # timeouts are not retried


def test_quota_error_returns_429(store, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 0)
    embedder = FakeEmbedder(DIM, failures=[RuntimeError("429 RESOURCE_EXHAUSTED")])

    with pytest.raises(RetrievalError) as err:
        search(store, embedder=embedder)
    assert err.value.status_code == 429


def test_rate_limited_query_is_retried(store):
    embedder = FakeEmbedder(DIM, failures=[FakeAPIError(429)], query_vector=[1.0, 0.0, 0.0])

    result = search(store, embedder=embedder)

    assert result["status"] == "ok"
    assert len(embedder.calls) == 2


# ---------- The endpoint ----------

@pytest.fixture
def client(store, monkeypatch):
    real_search = search_chunks

    def fake_search(query, top_k=None, document_id=None, min_similarity=None):
        return real_search(query, top_k=top_k, document_id=document_id,
                           min_similarity=min_similarity,
                           embedder=FakeEmbedder(DIM, query_vector=[1.0, 0.0, 0.0]),
                           store=store, sleep=no_sleep)
    monkeypatch.setattr(main, "search_chunks", fake_search)
    return TestClient(main.app)


def test_search_endpoint(client):
    response = client.post("/search", json={"query": "What was the revenue?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["results"][0]["filename"] == "Annual-Report.pdf"
    assert body["results"][0]["page_number"] == 6


def test_search_endpoint_validates_input(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "min_similarity": 2}).status_code == 422


def test_search_endpoint_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")

    response = TestClient(main.app).post("/search", json={"query": "revenue"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


# ---------- The real ChunkStore, with a mocked database connection ----------

def make_mock_connection(rows=(), count=0):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = list(rows)
    conn.execute.return_value.fetchone.return_value = (count,)
    return conn


def executed(conn):
    return [(str(c.args[0]), c.args[1] if len(c.args) > 1 else None)
            for c in conn.execute.call_args_list]


def test_store_search_uses_cosine_distance_and_the_limit():
    row = ("doc_p6_c1", "doc", "Revenue totaled...", "a.pdf", 6, 10, {"page": 6},
           "text", None, 6, None, None, 0.25)
    conn = make_mock_connection(rows=[row])

    results = ChunkStore(connect=lambda: conn).search([0.6, 0.8], limit=5)

    conn.transaction.assert_called_once()
    (settings_sql, settings_params), (sql, params) = executed(conn)
    assert "hnsw.ef_search" in settings_sql and "hnsw.iterative_scan" in settings_sql
    assert settings_params == ("40", "strict_order")
    assert "embedding <=> %(query)s::vector" in sql
    assert "order by embedding <=> %(query)s::vector" in sql
    assert "where embedding is not null" in sql
    assert "document_id = " not in sql
    assert params == {"query": "[0.6,0.8]", "limit": 5}
    assert results == [{
        "chunk_id": "doc_p6_c1", "document_id": "doc", "content": "Revenue totaled...",
        "source_filename": "a.pdf", "page_number": 6, "chunk_index": 10,
        "metadata": {"page": 6}, "chunk_type": "text", "section_path": None, "page_end": 6,
        "printed_page": None, "parent_chunk_id": None, "distance": 0.25,
    }]


def test_store_search_filters_by_document_and_raises_ef_search_for_large_k():
    conn = make_mock_connection()

    ChunkStore(connect=lambda: conn).search([1.0], limit=50, document_id="doc")

    (_, settings_params), (sql, params) = executed(conn)
    assert settings_params == ("50", "strict_order")
    assert "and document_id = %(document_id)s" in sql
    assert params["document_id"] == "doc"


def test_store_status_reads_the_setup_and_the_document_in_one_connection():
    conn = MagicMock()
    conn.__enter__.return_value = conn
    document_row = ("doc", "a.pdf", "visual", "ready", None, 12, 40, None, None)
    conn.execute.return_value.fetchone.side_effect = [(1536,), (True,), document_row]
    connections = []

    status = ChunkStore(connect=lambda: connections.append(1) or conn).status("doc")

    assert len(connections) == 1
    assert status["embedding_dimension"] == 1536
    assert status["migrated"] is True
    assert status["document"]["profile"] == "visual"
    assert status["document"]["chunk_count"] == 40


def test_store_status_without_the_table_skips_the_other_checks():
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = None

    status = ChunkStore(connect=lambda: conn).status("doc")

    assert status == {"embedding_dimension": None, "migrated": False, "document": None}
    assert conn.execute.call_count == 1


def test_store_counts_embedded_chunks():
    conn = make_mock_connection(count=7)
    store = ChunkStore(connect=lambda: conn)

    assert store.count_embedded_chunks() == 7
    assert store.count_embedded_chunks("doc") == 7

    (all_sql, all_params), (doc_sql, doc_params) = executed(conn)
    assert "embedding is not null" in all_sql and all_params == ()
    assert "document_id = %s" in doc_sql and doc_params == ("doc",)
