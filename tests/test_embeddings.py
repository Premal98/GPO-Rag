"""Tests for the embedding service. No real API calls are made."""

import math

import httpx
import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app import config, embeddings
from app.embeddings import (
    EmbeddingError,
    EmbeddingNotConfiguredError,
    embed_query,
    embed_texts,
    format_document_for_embedding,
    format_query_for_embedding,
    is_embedding_configured,
)
from fakes import FakeEmbedder


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error from API")
        self.code = code


def no_sleep(seconds):
    pass


@pytest.mark.parametrize("key", ["", "  ", "YOUR_GOOGLE_API_KEY"])
def test_placeholder_keys_are_not_configured(key):
    assert is_embedding_configured(key) is False


def test_create_embedder_refuses_placeholder_key(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    with pytest.raises(EmbeddingNotConfiguredError):
        embeddings.create_embedder()


def test_create_embedder_uses_settings(monkeypatch):
    # Creating the client does not call the API
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")

    embedder = embeddings.create_embedder()

    assert isinstance(embedder, GoogleGenerativeAIEmbeddings)
    assert embedder.model == config.EMBEDDING_MODEL
    assert embedder.output_dimensionality == config.EMBEDDING_DIMENSION
    timeout_ms = embedder.client._api_client._http_options.timeout
    assert timeout_ms == int(config.EMBEDDING_TIMEOUT_SECONDS * 1000)


def test_timeouts_are_detected_through_wrapped_errors():
    try:
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as inner:
            raise EmbeddingError("Embedding request failed") from inner
    except EmbeddingError as wrapped:
        assert embeddings.is_timeout(wrapped)
    assert not embeddings.is_timeout(EmbeddingError("API key not valid"))


def test_gemini_embedding_2_gets_document_prefix():
    text = format_document_for_embedding("Revenue grew 12%.", "annual-report", "gemini-embedding-2")
    assert text == "title: annual-report | text: Revenue grew 12%."


def test_older_model_gets_plain_text():
    assert format_document_for_embedding("Hi", "doc", "gemini-embedding-001") == "Hi"


def test_gemini_embedding_2_gets_query_prefix():
    text = format_query_for_embedding("What was revenue?", "gemini-embedding-2")
    assert text == "task: search result | query: What was revenue?"


def test_older_model_gets_plain_query():
    assert format_query_for_embedding("Hi", "gemini-embedding-001") == "Hi"


def test_embed_query_returns_a_normalized_vector():
    vector = embed_query(FakeEmbedder(dimension=8), "revenue", expected_dimension=8, sleep=no_sleep)

    assert len(vector) == 8
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)


def test_embed_query_rejects_wrong_dimension():
    with pytest.raises(EmbeddingError, match="768-dimension"):
        embed_query(FakeEmbedder(dimension=768), "a", expected_dimension=1536, sleep=no_sleep)


def test_texts_are_sent_in_batches():
    embedder = FakeEmbedder(dimension=8)
    texts = [f"text {i}" for i in range(23)]

    vectors = embed_texts(embedder, texts, expected_dimension=8, batch_size=10, sleep=no_sleep)

    assert [len(call) for call in embedder.calls] == [10, 10, 3]
    assert len(vectors) == 23


def test_vectors_are_normalized():
    vectors = embed_texts(FakeEmbedder(dimension=8), ["a", "bb"], 8, batch_size=10, sleep=no_sleep)

    for vector in vectors:
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)


def test_wrong_dimension_is_rejected():
    with pytest.raises(EmbeddingError, match="768-dimension"):
        embed_texts(FakeEmbedder(dimension=768), ["a"], expected_dimension=1536,
                    batch_size=10, sleep=no_sleep)


def test_rate_limit_is_retried_with_backoff():
    waits = []
    embedder = FakeEmbedder(dimension=8, failures=[FakeAPIError(429), FakeAPIError(503)])

    vectors = embed_texts(embedder, ["a"], 8, batch_size=10, sleep=waits.append)

    assert len(vectors) == 1
    assert waits == [2, 4]
    assert len(embedder.calls) == 3


# Shortened copy of a real free-tier error from the Gemini API
FREE_TIER_429 = (
    "Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
    "'message': 'You exceeded your current quota. \\n* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/embed_content_free_tier_requests, limit: 100, "
    "model: gemini-embedding-2\\nPlease retry in 50.668013062s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'quotaId': 'EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier'}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '50s'}]}}"
)


def test_retry_delay_is_read_from_the_error():
    assert embeddings.suggested_retry_delay(RuntimeError(FREE_TIER_429)) == pytest.approx(50.668013062)
    assert embeddings.suggested_retry_delay(RuntimeError("'retryDelay': '12s'")) == 12
    assert embeddings.suggested_retry_delay(RuntimeError("API key not valid")) is None


def test_rate_limit_waits_as_long_as_gemini_asks():
    waits = []
    embedder = FakeEmbedder(dimension=8, failures=[RuntimeError(FREE_TIER_429)])

    vectors = embed_texts(embedder, ["a"], 8, batch_size=10, sleep=waits.append)

    assert len(vectors) == 1
    assert waits == [pytest.approx(51.668013062)]  # the suggested delay plus a second


def test_rate_limit_with_a_very_long_delay_gives_up(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_RETRY_WAIT_SECONDS", 30)
    waits = []
    embedder = FakeEmbedder(dimension=8, failures=[RuntimeError(FREE_TIER_429)])

    with pytest.raises(EmbeddingError, match="EMBEDDING_MAX_RETRY_WAIT_SECONDS"):
        embed_texts(embedder, ["a"], 8, batch_size=10, sleep=waits.append)
    assert waits == []


def test_daily_quota_is_not_retried():
    error = RuntimeError("429 RESOURCE_EXHAUSTED 'quotaId': 'EmbedContentRequestsPerDayPerProject'")
    embedder = FakeEmbedder(dimension=8, failures=[error])

    with pytest.raises(EmbeddingError, match="daily"):
        embed_texts(embedder, ["a"], 8, batch_size=10, sleep=no_sleep)
    assert len(embedder.calls) == 1


def test_rate_limit_detection():
    assert embeddings.is_rate_limited(EmbeddingError(FREE_TIER_429))
    assert not embeddings.is_rate_limited(EmbeddingError("API key not valid"))


def test_gives_up_after_max_retries():
    failures = [FakeAPIError(429)] * 3
    embedder = FakeEmbedder(dimension=8, failures=failures)

    with pytest.raises(EmbeddingError, match="failed"):
        embed_texts(embedder, ["a"], 8, batch_size=10, max_retries=2, sleep=no_sleep)
    assert len(embedder.calls) == 3


def test_permanent_errors_are_not_retried():
    embedder = FakeEmbedder(dimension=8, failures=[FakeAPIError(400)])

    with pytest.raises(EmbeddingError):
        embed_texts(embedder, ["a"], 8, batch_size=10, sleep=no_sleep)
    assert len(embedder.calls) == 1


def test_wrapped_langchain_error_is_detected_as_retryable():
    # LangChain wraps Google errors; the original error is kept as __cause__
    try:
        try:
            raise FakeAPIError(429)
        except FakeAPIError as inner:
            raise RuntimeError("Error embedding content") from inner
    except RuntimeError as wrapped:
        assert embeddings._is_retryable(wrapped)
