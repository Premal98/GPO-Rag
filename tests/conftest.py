"""Shared test setup.

Unit tests must never use real credentials, even when a real .env exists:
they would spend API quota or write to your database. This replaces both
secrets with placeholders for every test not marked "integration".
"""

import pytest

from app import config, pipeline, retrieval
from fakes import FakeEnricher, FakeLayoutParser


@pytest.fixture(autouse=True)
def placeholder_credentials(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
    monkeypatch.setattr(
        config, "SUPABASE_DATABASE_URL",
        "postgresql://postgres.YOUR_PROJECT_REF:YOUR_DB_PASSWORD@YOUR_POOLER_HOST:5432/postgres",
    )


@pytest.fixture(autouse=True)
def layout_parser(request, monkeypatch):
    """Replace Docling with a fast fake in unit tests. Tests can inspect or configure it."""
    if request.node.get_closest_marker("integration"):
        return None
    fake = FakeLayoutParser()
    monkeypatch.setattr(pipeline, "parse_pdf", fake)
    return fake


@pytest.fixture(autouse=True)
def enricher(request, monkeypatch):
    """Replace Gemini enrichment with a fake in unit tests, so no test can spend quota."""
    if request.node.get_closest_marker("integration"):
        return None
    fake = FakeEnricher()
    monkeypatch.setattr(pipeline, "enrich_layout", fake)
    return fake


@pytest.fixture(autouse=True)
def no_reranker(request, monkeypatch):
    """Unit tests never load the reranker model; tests that need one pass a fake."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr(retrieval, "get_reranker", lambda: None)
