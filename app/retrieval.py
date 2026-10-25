"""Hybrid search: finds the stored chunks most relevant to a question.

The question is embedded with the same model, dimension, and normalization
used at ingestion and compared to stored chunks by cosine distance in
pgvector; at the same time Postgres full-text search finds chunks sharing
its words (exact names, years, codes). Both lists are merged, reranked by a
local cross-encoder, and expanded (parent sections, page footnotes). No
answer is generated here; this only returns the chunks.
"""

import re
import time

import psycopg

from app import config
from app.database import MIGRATION_002, is_database_configured
from app.embeddings import (
    EmbeddingError,
    create_embedder,
    embed_query,
    format_query_for_embedding,
    is_embedding_configured,
    is_rate_limited,
    is_timeout,
)
from app.profiles import profile_setting
from app.rerank import get_reranker
from app.vector_store import BUSY_STATUSES, ChunkStore

METRIC = "cosine"


class RetrievalError(Exception):
    """A problem we report back to the user, with an HTTP status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


RRF_K = 60  # standard constant for reciprocal rank fusion
MAX_KEYWORD_TERMS = 24
PAGE_FOOTNOTES_MAX_CHARS = 1500

# "chapter 3", "section 2.1", "Box 1.2", "annex A", "Part II" in a question.
# The kind may be in any case, but letters and roman numerals must be upper
# case, so "which section a user reads" or "the part I liked" don't count
# (a lone letter only when punctuation or the end follows: "What does Part I say?" is
# ambiguous and left out).
SECTION_REFERENCE = re.compile(
    r"\b(?i:(chapter|section|part|box|annex|appendix))\s+"
    r"(\d+(?:\.\d+)*|[A-Z](?![A-Za-z])|[IVXLC]{2,}\b|[IVX](?=\s*(?:[.:,;?!)]|$)))"
)


def keyword_terms(query):
    """Plain words for keyword search (letters and digits only, so safe in to_tsquery)."""
    terms = []
    for word in re.findall(r"[a-z0-9]+", query.lower()):
        if (len(word) > 1 or word.isdigit()) and word not in terms:
            terms.append(word)
    return terms[:MAX_KEYWORD_TERMS]


def section_pattern(query):
    """A Postgres regex for section_path if the question names a section, else None.

    "chapter 3" matches paths with a heading like "3 Outlook", "3. Outlook",
    "3.2 Prices" or "Chapter 3 ...". (Postgres regexes have no \\b, so word
    edges are written out.)
    """
    match = SECTION_REFERENCE.search(query)
    if not match:
        return None
    kind, number = match.group(1).lower(), re.escape(match.group(2))
    following = query[match.end():].lstrip()[:1]
    if len(match.group(2)) == 1 and match.group(2).isalpha() \
            and kind not in ("annex", "appendix") and following not in ("", *".:,;?!)"):
        return None  # "the part I liked": a lone letter here is usually a word
    if kind in ("chapter", "section", "part"):
        return rf"(^|> )({kind}\s+)?{number}([.:\s]|$)"
    return rf"(^|[^a-z0-9]){kind}\s+{number}([^a-z0-9]|$)"


def fuse(*rankings):
    """Merge ranked lists of rows by reciprocal rank fusion, best first.

    A row's score is the sum of 1 / (RRF_K + rank) over the lists it is in,
    so chunks found by both vector and keyword search rise to the top.
    """
    scores, rows = {}, {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            scores[row["chunk_id"]] = scores.get(row["chunk_id"], 0) + 1 / (RRF_K + rank)
            rows.setdefault(row["chunk_id"], row)
    return sorted(rows.values(), key=lambda row: scores[row["chunk_id"]], reverse=True)


def _to_result(rank, row, matched=None, rerank_score=None):
    """A result for the response. matched is the child chunk a parent was found through."""
    distance = float((matched or row)["distance"])
    return {
        "rank": rank,
        "chunk_id": row["chunk_id"],
        "document_id": row["document_id"],
        "filename": row["source_filename"],
        "page_number": row["page_number"],
        "page_end": row.get("page_end") or row["page_number"],
        "printed_page": row.get("printed_page"),
        "chunk_index": row["chunk_index"],
        "chunk_type": row.get("chunk_type") or "text",
        "section_path": row.get("section_path"),
        "matched_chunk_id": matched["chunk_id"] if matched else row["chunk_id"],
        # For normalized vectors, cosine similarity = 1 - cosine distance.
        # 1 means the same meaning, around 0 means unrelated.
        "similarity": round(1 - distance, 6),
        "distance": round(distance, 6),
        "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
        "content": row["content"],
    }


def _select(rows, scores, top_k, store, document_id, profile):
    """Pick the final chunks from ranked candidates.

    Child chunks are replaced by their parent section (once per parent), and
    chunks are added until top_k or RETRIEVAL_CONTEXT_MAX_CHARS is reached
    (the first chunk always fits). Visual documents get their page footnotes.
    """
    scores = scores or [None] * len(rows)
    # Parents of the candidates that can still be picked, fetched in one query
    parent_ids = {r["parent_chunk_id"] for r in rows[:top_k * 4] if r.get("parent_chunk_id")}
    parents = store.get_chunks(parent_ids) if parent_ids else {}

    results, seen, total = [], set(), 0
    for row, score in zip(rows, scores):
        parent = parents.get(row.get("parent_chunk_id"))
        key = parent["chunk_id"] if parent else row["chunk_id"]
        if key in seen:
            continue
        chosen = parent or row
        if results and total + len(chosen["content"]) > config.RETRIEVAL_CONTEXT_MAX_CHARS:
            continue  # a shorter chunk further down may still fit
        seen.add(key)
        total += len(chosen["content"])
        results.append(_to_result(len(results) + 1, chosen, row if parent else None, score))
        if len(results) == top_k:
            break

    if document_id and profile_setting(profile, "footnote_chunks"):
        pages = {r["page_number"] for r in results if r["chunk_type"] == "text"}
        footnotes = store.get_footnotes(document_id, pages) if pages else {}
        for result in results:
            notes = footnotes.get(result["page_number"]) if result["chunk_type"] == "text" else None
            if notes:
                text = "\n".join(f"- {n}" for n in notes)[:PAGE_FOOTNOTES_MAX_CHARS]
                result["content"] += f"\n\nFootnotes on this page:\n{text}"
    return results


def _model_warnings(rows):
    """Warn if matched chunks were embedded with a different model than queries use.

    Vectors from different models live in different spaces, so their
    similarity scores are meaningless even if the dimensions match.
    """
    other_models = sorted({
        (row["metadata"] or {}).get("embedding_model") or "unknown"
        for row in rows
        if (row["metadata"] or {}).get("embedding_model") != config.EMBEDDING_MODEL
    })
    if not other_models:
        return []
    return [
        f"Some matched chunks were embedded with {', '.join(other_models)}, but queries use "
        f"{config.EMBEDDING_MODEL}. Re-ingest those documents with force=true for reliable results."
    ]


def search_chunks(query, top_k=None, document_id=None, min_similarity=None,
                  embedder=None, store=None, sleep=time.sleep, reranker=None):
    """Find the top_k chunks most relevant to the query.

    Steps: embed the question; take RETRIEVAL_CANDIDATES candidates each from
    vector search and keyword search (only in the named section if the
    question names one, e.g. "in chapter 3"); merge them by reciprocal rank
    fusion; drop those below min_similarity; rerank them with a local
    cross-encoder; then pick the best, replacing small child chunks with their
    parent section, until top_k chunks or RETRIEVAL_CONTEXT_MAX_CHARS of text.
    For visual documents, footnotes on a chunk's page are added to it.

    top_k defaults to the document profile's top_k (RETRIEVAL_TOP_K when
    searching all documents); min_similarity defaults to
    RETRIEVAL_MIN_SIMILARITY (None means no cutoff). document_id limits the
    search to one document. embedder, store, sleep and reranker can be
    replaced in tests.

    Returns a dictionary whose "status" is:
      "ok"             - results were found
      "empty_database" - no chunks have been ingested (for that document, if given)
      "no_matches"     - chunks exist, but none reached min_similarity
    """
    query = (query or "").strip()
    if not query:
        raise RetrievalError("The query is empty.", 422)
    if top_k is not None and not 1 <= top_k <= config.RETRIEVAL_MAX_TOP_K:
        raise RetrievalError(f"top_k must be between 1 and {config.RETRIEVAL_MAX_TOP_K}.", 422)
    if min_similarity is None:
        min_similarity = config.RETRIEVAL_MIN_SIMILARITY
    if min_similarity is not None and not -1 <= min_similarity <= 1:
        raise RetrievalError("min_similarity must be between -1 and 1.", 422)

    # Check settings before doing anything that costs time or API quota
    if embedder is None:
        if not is_embedding_configured():
            raise RetrievalError("GOOGLE_API_KEY is not set in .env.", 503)
        embedder = create_embedder()
    if store is None:
        if not is_database_configured():
            raise RetrievalError("SUPABASE_DATABASE_URL is not set in .env.", 503)
        store = ChunkStore()

    try:
        setup = store.status(document_id)
    except psycopg.Error as err:
        raise RetrievalError(f"Database error: {str(err).strip()}", 503) from err
    column_dimension = setup["embedding_dimension"]
    if column_dimension is None:
        raise RetrievalError(
            "The document_chunks table does not exist. Run sql/001_create_document_chunks.sql.", 503
        )
    if column_dimension != config.EMBEDDING_DIMENSION:
        raise RetrievalError(
            f"The database embedding column has {column_dimension} dimensions, but "
            f"EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}. They must match.", 409
        )
    if not setup["migrated"]:
        raise RetrievalError(f"The documents table is missing. Run {MIGRATION_002}.", 503)
    # Checked before embedding the question, so a wrong id costs no API quota
    if document_id is not None and setup["document"] is None:
        raise RetrievalError(
            f"Document '{document_id}' is not in the database. GET /documents lists the "
            f"available documents; if you just uploaded it, call POST /documents/{document_id}/ingest.",
            404,
        )
    document = setup["document"]
    if document is not None and document["status"] != "ready":
        if document["status"] in BUSY_STATUSES:
            raise RetrievalError(
                f"Document '{document_id}' is still being processed (status: {document['status']}). "
                f"Try again when GET /documents/{document_id} shows status ready.", 409
            )
        raise RetrievalError(
            f"Processing document '{document_id}' failed: {document['status_message'] or 'unknown error'} "
            "Upload it again with POST /documents.", 409
        )

    started = time.perf_counter()
    try:
        query_vector = embed_query(
            embedder,
            format_query_for_embedding(query, config.EMBEDDING_MODEL),
            expected_dimension=config.EMBEDDING_DIMENSION,
            max_retries=config.EMBEDDING_MAX_RETRIES,
            sleep=sleep,
        )
    except EmbeddingError as err:
        if is_timeout(err):
            raise RetrievalError(
                f"Embedding the question timed out after {config.EMBEDDING_TIMEOUT_SECONDS:g} "
                "seconds. Try again, or raise EMBEDDING_TIMEOUT_SECONDS.", 504
            ) from err
        if is_rate_limited(err):
            raise RetrievalError(
                f"The Gemini embedding quota was reached. Wait a minute and try again. Details: {err}",
                429,
            ) from err
        raise RetrievalError(str(err), 502) from err
    embedded = time.perf_counter()

    profile = document["profile"] if document else None
    if top_k is None:
        top_k = min(profile_setting(profile, "top_k", config.RETRIEVAL_TOP_K) if profile
                    else config.RETRIEVAL_TOP_K, config.RETRIEVAL_MAX_TOP_K)
    limit = max(config.RETRIEVAL_CANDIDATES, top_k)
    terms = keyword_terms(query)
    pattern = section_pattern(query)
    try:
        vector_rows, keyword_rows = store.search_candidates(
            query_vector, terms, limit, document_id=document_id, section_pattern=pattern)
        if pattern and not (vector_rows or keyword_rows):
            # The question names a section the document doesn't have: search everything
            pattern = None
            vector_rows, keyword_rows = store.search_candidates(
                query_vector, terms, limit, document_id=document_id)
        rows = fuse(vector_rows, keyword_rows)
        # Only count when nothing came back, to tell "empty" from "no good match"
        available = store.count_embedded_chunks(document_id) if not rows else None
    except psycopg.Error as err:
        raise RetrievalError(f"Database error while searching: {str(err).strip()}", 503) from err
    searched = time.perf_counter()

    if min_similarity is not None:
        rows = [r for r in rows if 1 - float(r["distance"]) >= min_similarity]

    warnings = _model_warnings(rows)
    reranker = get_reranker() if reranker is None else reranker
    scores = None
    if reranker and len(rows) > 1:
        # The best candidates are reranked; the rest keep their hybrid order after them
        head, tail = rows[:config.RERANK_CANDIDATES], rows[config.RERANK_CANDIDATES:]
        try:
            head_scores = reranker.score(query, [r["content"] for r in head])
        except Exception as err:  # e.g. the model could not be downloaded
            warnings.append(f"Reranking was skipped ({err}); results are in hybrid search order.")
        else:
            ranked = sorted(zip(head_scores, head), key=lambda pair: -pair[0])
            rows = [r for _, r in ranked] + tail
            scores = [score for score, _ in ranked] + [None] * len(tail)
    reranked = time.perf_counter()

    try:
        results = _select(rows, scores, top_k, store, document_id, profile)
    except psycopg.Error as err:
        raise RetrievalError(f"Database error while reading chunks: {str(err).strip()}", 503) from err

    if available == 0:
        status = "empty_database"
        message = ("This document has no embedded chunks. Ingest it with "
                   f"POST /documents/{document_id}/ingest." if document_id else
                   "The database has no embedded chunks yet. Upload a PDF and ingest it first.")
    elif not results:
        status = "no_matches"
        message = (f"No chunks reached the minimum similarity of {min_similarity}. "
                   "Try rephrasing the question or lowering min_similarity.")
    else:
        status = "ok"
        message = f"Found {len(results)} relevant chunks."

    return {
        "status": status,
        "message": message,
        "query": query,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "document_id": document_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "metric": METRIC,
        "profile": profile,
        "mode": "hybrid",
        "candidates": len(rows),
        "reranked": scores is not None,
        "section_filter": pattern,
        "result_count": len(results),
        "timing_ms": {
            "embedding": round((embedded - started) * 1000, 1),
            "search": round((searched - embedded) * 1000, 1),
            "rerank": round((reranked - searched) * 1000, 1),
        },
        "warnings": warnings,
        "results": results,
    }
