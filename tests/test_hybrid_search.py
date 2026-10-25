"""Tests for hybrid search, reranking, parent expansion, footnotes and section filters."""

from unittest.mock import MagicMock

import pytest

from app import config
from app.retrieval import fuse, keyword_terms, search_chunks, section_pattern
from app.vector_store import ChunkStore
from fakes import FakeChunkStore, FakeEmbedder

DIM = 3


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", DIM)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setattr(config, "RETRIEVAL_TOP_K", 3)
    monkeypatch.setattr(config, "RETRIEVAL_MAX_TOP_K", 20)
    monkeypatch.setattr(config, "RETRIEVAL_MIN_SIMILARITY", None)
    monkeypatch.setattr(config, "RETRIEVAL_CANDIDATES", 10)
    monkeypatch.setattr(config, "RETRIEVAL_CONTEXT_MAX_CHARS", 12000)


def add(store, chunk_id, embedding, content, page=1, document_id="doc1", **extra):
    store.rows[chunk_id] = {
        "chunk_id": chunk_id, "document_id": document_id, "content": content,
        "source_filename": "report.pdf", "page_number": page, "chunk_index": len(store.rows),
        "metadata": {"embedding_model": "gemini-embedding-2"}, "embedding": embedding, **extra,
    }


def search(store, query="question", reranker=None, **kwargs):
    return search_chunks(query, embedder=FakeEmbedder(DIM, query_vector=[1.0, 0.0, 0.0]),
                         store=store, sleep=lambda s: None, reranker=reranker, **kwargs)


class FakeReranker:
    """Scores texts by a fixed table; unknown texts score 0."""

    def __init__(self, scores=None, error=None):
        self.scores = scores or {}
        self.error = error
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        if self.error:
            raise self.error
        return [self.scores.get(text, 0.0) for text in texts]


# ---------- Building blocks ----------

def test_keyword_terms_are_plain_unique_words():
    assert keyword_terms("What was GPO's revenue in FY 2018? Revenue!") == [
        "what", "was", "gpo", "revenue", "in", "fy", "2018"]
    assert keyword_terms("a ; -- ' DROP") == ["drop"]  # nothing that could break to_tsquery


@pytest.mark.parametrize("question, matches, does_not_match", [
    ("What does chapter 3 say?", ["3 Outlook", "Chapter 3 Outlook", "Report > 3. Prices", "3.2 Food"],
     ["13 Other", "2 Outlook", "Annex 3"]),
    ("Summarize section 2.1", ["2 Outlook > 2.1 Inflation", "Section 2.1 Inflation"], ["2.10 Debt", "2 Outlook"]),
    ("What is in Box 1.2?", ["1 Outlook > Box 1.2 Commodity prices"], ["Box 1.21 Other", "1.2 Prices"]),
    ("Explain annex A", ["Annex A Methods"], ["Annex B Data"]),
])
def test_section_patterns(question, matches, does_not_match):
    import re
    pattern = section_pattern(question)

    assert all(re.search(pattern, path, re.IGNORECASE) for path in matches)
    assert not any(re.search(pattern, path, re.IGNORECASE) for path in does_not_match)


def test_questions_without_a_section_reference_have_no_filter():
    assert section_pattern("How did growth change in 2024?") is None


def test_fusion_favours_chunks_found_by_both_searches():
    a, b, c = ({"chunk_id": x} for x in "abc")

    assert [r["chunk_id"] for r in fuse([a, b], [c, b])] == ["b", "a", "c"]


# ---------- Hybrid search ----------

def test_keyword_search_finds_exact_terms_vector_search_ranks_low():
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "close", [1.0, 0.1, 0.0], "General discussion of the economy.")
    add(store, "exact", [0.0, 1.0, 0.3], "Scope 3 emissions were 16.1 million tonnes.")

    result = search(store, query="Scope 3 emissions", top_k=2)

    assert [r["chunk_id"] for r in result["results"]] == ["exact", "close"]
    assert result["mode"] == "hybrid"
    assert store.keyword_searches[0][0] == ["scope", "3", "emissions"]


def test_min_similarity_applies_to_keyword_results_too():
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "exact", [0.0, 1.0, 0.0], "Scope 3 emissions.")  # similarity 0

    result = search(store, query="Scope 3", min_similarity=0.5)

    assert result["status"] == "no_matches"


# ---------- Reranking ----------

def test_reranker_reorders_candidates():
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "a", [1.0, 0.0, 0.0], "Text A")
    add(store, "b", [1.0, 0.5, 0.0], "Text B")
    reranker = FakeReranker({"Text A": 0.1, "Text B": 2.5})

    result = search(store, reranker=reranker, top_k=2)

    assert [r["chunk_id"] for r in result["results"]] == ["b", "a"]
    assert [r["rerank_score"] for r in result["results"]] == [2.5, 0.1]
    assert result["reranked"] is True
    assert reranker.calls[0][0] == "question"


def test_reranker_failure_falls_back_to_hybrid_order():
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "a", [1.0, 0.0, 0.0], "Text A")
    add(store, "b", [1.0, 0.5, 0.0], "Text B")

    result = search(store, reranker=FakeReranker(error=OSError("model download failed")))

    assert [r["chunk_id"] for r in result["results"]] == ["a", "b"]
    assert result["reranked"] is False
    assert "Reranking was skipped" in result["warnings"][0]


# ---------- Profiles, parents, budget, footnotes ----------

def test_top_k_defaults_to_the_document_profile():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="visual")
    for n in range(12):
        add(store, f"c{n}", [1.0, n / 10, 0.0], f"Chunk {n}")

    result = search(store, document_id="doc1")

    assert result["profile"] == "visual"
    assert result["top_k"] == 8 and result["result_count"] == 8


def test_child_matches_are_replaced_by_their_parent_once():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="structured")
    add(store, "p1", None, "Whole section about inflation.", page=4, page_end=6,
        section_path="2 Outlook > 2.1 Inflation")
    add(store, "p1_c1", [1.0, 0.0, 0.0], "about inflation", page=4, parent_chunk_id="p1")
    add(store, "p1_c2", [1.0, 0.1, 0.0], "section about", page=4, parent_chunk_id="p1")
    add(store, "other", [1.0, 0.3, 0.0], "Another chunk.", page=9)

    result = search(store, document_id="doc1", top_k=3)

    first = result["results"][0]
    assert first["chunk_id"] == "p1"
    assert first["content"] == "Whole section about inflation."
    assert (first["page_number"], first["page_end"]) == (4, 6)
    assert first["section_path"] == "2 Outlook > 2.1 Inflation"
    assert first["matched_chunk_id"] == "p1_c1"
    assert [r["chunk_id"] for r in result["results"]] == ["p1", "other"]


def test_context_budget_limits_how_much_text_is_returned(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_CONTEXT_MAX_CHARS", 1000)
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "big1", [1.0, 0.0, 0.0], "x" * 700)
    add(store, "big2", [1.0, 0.1, 0.0], "y" * 700)
    add(store, "small", [1.0, 0.2, 0.0], "z" * 200)

    result = search(store, top_k=3)

    assert [r["chunk_id"] for r in result["results"]] == ["big1", "small"]


def test_first_chunk_is_returned_even_if_it_exceeds_the_budget(monkeypatch):
    monkeypatch.setattr(config, "RETRIEVAL_CONTEXT_MAX_CHARS", 1000)
    store = FakeChunkStore(DIM)
    store.register("doc1")
    add(store, "huge", [1.0, 0.0, 0.0], "x" * 5000)

    assert search(store)["result_count"] == 1


def test_visual_documents_get_footnotes_of_the_page():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="visual")
    add(store, "text", [1.0, 0.0, 0.0], "Emissions fell 60 percent.", page=3)
    add(store, "fn", [0.0, 0.0, 1.0], "Footnote: Compared with fiscal 2015.", page=3,
        chunk_type="footnote")

    result = search(store, document_id="doc1", query="q", top_k=1)

    assert result["results"][0]["content"] == (
        "Emissions fell 60 percent.\n\nFootnotes on this page:\n- Footnote: Compared with fiscal 2015.")


def test_narrative_documents_get_no_footnotes_added():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="narrative")
    add(store, "text", [1.0, 0.0, 0.0], "Revenue rose.", page=3)
    add(store, "fn", [0.0, 0.0, 1.0], "Footnote: Unaudited.", page=3, chunk_type="footnote")

    result = search(store, document_id="doc1", query="q", top_k=1)

    assert result["results"][0]["content"] == "Revenue rose."


# ---------- Section filter ----------

def test_question_naming_a_chapter_searches_only_that_chapter():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="structured")
    add(store, "ch2", [1.0, 0.0, 0.0], "Growth text.", section_path="2 Outlook")
    add(store, "ch3", [0.5, 0.5, 0.0], "Risk text.", section_path="3 Risks > 3.1 Debt")

    result = search(store, query="What does chapter 3 say?", document_id="doc1")

    assert [r["chunk_id"] for r in result["results"]] == ["ch3"]
    assert result["section_filter"] is not None


def test_unknown_section_falls_back_to_the_whole_document():
    store = FakeChunkStore(DIM)
    store.register("doc1", profile="structured")
    add(store, "ch2", [1.0, 0.0, 0.0], "Growth text.", section_path="2 Outlook")

    result = search(store, query="What does chapter 9 say?", document_id="doc1")

    assert [r["chunk_id"] for r in result["results"]] == ["ch2"]
    assert result["section_filter"] is None


# ---------- The real ChunkStore, with a mocked database connection ----------

def mock_connection(fetchall=()):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = list(fetchall)
    return conn


def test_candidates_run_vector_and_keyword_search_in_one_connection():
    conn = mock_connection()
    connections = []

    ChunkStore(connect=lambda: connections.append(1) or conn).search_candidates(
        [1.0], ["scope", "3"], 30, document_id="doc", section_pattern="(^|> )3")

    assert len(connections) == 1
    sqls = [(str(c.args[0]), c.args[1] if len(c.args) > 1 else None)
            for c in conn.execute.call_args_list]
    keyword_sql, keyword_params = sqls[-1]
    assert "content_tsv @@ to_tsquery('english', %(terms)s)" in keyword_sql
    assert "ts_rank_cd" in keyword_sql and "embedding is not null" in keyword_sql
    assert "section_path ~* %(section)s" in keyword_sql
    assert keyword_params["terms"] == "scope | 3"
    assert keyword_params["document_id"] == "doc"


def test_no_keyword_query_without_terms():
    conn = mock_connection()

    vector, keyword = ChunkStore(connect=lambda: conn).search_candidates([1.0], [], 30)

    assert keyword == []
    assert not any("to_tsquery" in str(c.args[0]) for c in conn.execute.call_args_list)


def test_only_the_best_candidates_are_reranked(monkeypatch):
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 2)
    store = FakeChunkStore(DIM)
    store.register("doc1")
    for n in range(4):
        add(store, f"c{n}", [1.0, n / 10, 0.0], f"Text {n}")
    reranker = FakeReranker({"Text 0": 0.1, "Text 1": 0.9})

    result = search(store, reranker=reranker, top_k=4)

    assert len(reranker.calls[0][1]) == 2
    assert [r["chunk_id"] for r in result["results"]] == ["c1", "c0", "c2", "c3"]
    assert [r["rerank_score"] for r in result["results"]] == [0.9, 0.1, None, None]


@pytest.mark.parametrize("question", ["Which section a user should read first?",
                                      "What was the part I liked most?",
                                      "What does Part I say?", "explain appendix b"])
def test_ordinary_words_are_not_mistaken_for_sections(question):
    assert section_pattern(question) is None


@pytest.mark.parametrize("question", ["Summarize Part II", "What is in Annex A?",
                                      "Summarize Part I.", "What does chapter 3 say?"])
def test_section_references_are_recognized(question):
    assert section_pattern(question) is not None
