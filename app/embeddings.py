"""Creates Gemini embeddings for chunks, in batches, with retries.

Uses LangChain's GoogleGenerativeAIEmbeddings. Nothing here calls the API
unless embed_texts() is called with a real embedder.
"""

import math
import re
import time

import httpx
from google import genai
from google.genai import types as genai_types
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app import config

# HTTP codes worth retrying: rate limit (429) and temporary server problems
RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_TEXT = ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "503")


class EmbeddingNotConfiguredError(Exception):
    """GOOGLE_API_KEY is missing or still a placeholder."""


class EmbeddingError(Exception):
    """Embedding failed, or the model returned something unexpected."""


def is_embedding_configured(api_key=None):
    key = (config.GOOGLE_API_KEY if api_key is None else api_key).strip()
    return bool(key) and "YOUR_" not in key


def create_embedder():
    """Build the LangChain embeddings client from the .env settings."""
    if not is_embedding_configured():
        raise EmbeddingNotConfiguredError(
            "GOOGLE_API_KEY is not set. Add your Gemini API key to the .env file."
        )
    embedder = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        output_dimensionality=config.EMBEDDING_DIMENSION,
    )
    # LangChain's embeddings class has no timeout setting, and without one a
    # stalled request waits forever. Swap in a Google client that has one.
    embedder.client = genai.Client(
        api_key=config.GOOGLE_API_KEY,
        http_options=genai_types.HttpOptions(timeout=int(config.EMBEDDING_TIMEOUT_SECONDS * 1000)),
    )
    return embedder


def format_document_for_embedding(text, title, model):
    """Add the prefix Google recommends for documents in retrieval tasks.

    gemini-embedding-2 does not use task_type; instead the task is written
    into the text. Older models (gemini-embedding-001) use task_type, which
    LangChain sets to RETRIEVAL_DOCUMENT automatically, so no prefix is needed.
    """
    if model.startswith("gemini-embedding-2"):
        return f"title: {title or 'none'} | text: {text}"
    return text


def format_query_for_embedding(text, model):
    """Add the prefix Google recommends for search queries.

    This is the query-side partner of format_document_for_embedding(). For
    older models, embed_query() makes LangChain send task_type RETRIEVAL_QUERY.
    """
    if model.startswith("gemini-embedding-2"):
        return f"task: search result | query: {text}"
    return text


def normalize(vector):
    """Scale a vector to length 1 so all stored embeddings are comparable."""
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        raise EmbeddingError("The model returned an all-zero embedding.")
    return [x / length for x in vector]


def is_timeout(error):
    """True if the error, or anything that caused it, was a request timeout."""
    while error is not None:
        if isinstance(error, (httpx.TimeoutException, TimeoutError)):
            return True
        if "DEADLINE_EXCEEDED" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def is_rate_limited(error):
    """True if Gemini refused the request because a quota was used up (HTTP 429)."""
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or re.search(r"\b429\b", text) is not None


def suggested_retry_delay(error):
    """Seconds Gemini asked us to wait before retrying, or None if it didn't say.

    Rate-limit errors include e.g. "Please retry in 50.67s" or 'retryDelay': '50s'.
    """
    match = re.search(r"retry in ([\d.]+)\s*s|'retryDelay': '([\d.]+)s'", str(error))
    return float(match.group(1) or match.group(2)) if match else None


def _is_retryable(error):
    cause = error.__cause__ or error
    if getattr(cause, "code", None) in RETRYABLE_CODES:
        return True
    return any(marker in str(error) for marker in RETRYABLE_TEXT)


def _call_with_retry(request, max_retries, sleep):
    for attempt in range(max_retries + 1):
        try:
            return request()
        except Exception as error:
            if not _is_retryable(error):
                raise EmbeddingError(f"Embedding request failed: {error}") from error
            if "PerDay" in str(error):
                # Waiting a few seconds won't help with a daily limit
                raise EmbeddingError(
                    "The daily Gemini embedding quota is used up. Try again tomorrow, or "
                    f"enable billing for higher limits. Details: {error}"
                ) from error

            wait = 2 ** (attempt + 1)  # wait 2s, 4s, 8s, ... before trying again
            suggested = suggested_retry_delay(error)
            if suggested is not None:
                # The free tier's per-minute quota resets after the delay Gemini gives
                if suggested > config.EMBEDDING_MAX_RETRY_WAIT_SECONDS:
                    raise EmbeddingError(
                        f"Gemini asked to wait {suggested:.0f}s before retrying, which is longer "
                        "than EMBEDDING_MAX_RETRY_WAIT_SECONDS. Try again later. "
                        f"Details: {error}"
                    ) from error
                wait = max(wait, suggested + 1)
            if attempt == max_retries:
                raise EmbeddingError(f"Embedding request failed: {error}") from error
            sleep(wait)


def _embed_batch_with_retry(embedder, texts, max_retries, sleep):
    return _call_with_retry(
        lambda: embedder.embed_documents(texts, batch_size=len(texts)), max_retries, sleep
    )


def _check_dimension(vector, expected_dimension):
    if len(vector) != expected_dimension:
        raise EmbeddingError(
            f"The model returned {len(vector)}-dimension embeddings, but "
            f"EMBEDDING_DIMENSION is {expected_dimension}. Check EMBEDDING_MODEL "
            "and EMBEDDING_DIMENSION in .env."
        )


def embed_query(embedder, text, expected_dimension, max_retries=4, sleep=time.sleep):
    """Embed one search query the same way stored chunks were embedded.

    The text should already be formatted with format_query_for_embedding().
    Returns a normalized vector, so cosine distance compares like with like.
    """
    vector = _call_with_retry(lambda: embedder.embed_query(text), max_retries, sleep)
    _check_dimension(vector, expected_dimension)
    return normalize(vector)


def embed_texts(embedder, texts, expected_dimension, batch_size, max_retries=4, sleep=time.sleep):
    """Embed texts in batches and check that every vector has the expected size.

    Returns a list of normalized vectors, one per text, in the same order.
    """
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_vectors = _embed_batch_with_retry(embedder, batch, max_retries, sleep)

        if len(batch_vectors) != len(batch):
            raise EmbeddingError(
                f"Asked for {len(batch)} embeddings but received {len(batch_vectors)}."
            )
        for vector in batch_vectors:
            _check_dimension(vector, expected_dimension)
            vectors.append(normalize(vector))
    return vectors
