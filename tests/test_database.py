"""Tests for the database setup.

The first tests never touch the network. The last one connects to your real
Supabase database, and is skipped unless SUPABASE_DATABASE_URL is set in .env.
"""

import re
from pathlib import Path

import pytest

from app import config, database
from app.database import DatabaseNotConfiguredError, is_database_configured

SQL_FILE = Path(__file__).parent.parent / "sql" / "001_create_document_chunks.sql"


@pytest.mark.parametrize("url", [
    "",
    "   ",
    "YOUR_SUPABASE_DATABASE_URL",
    "postgresql://postgres.YOUR_PROJECT_REF:YOUR_DB_PASSWORD@YOUR_POOLER_HOST:5432/postgres",
    "mysql://user:pass@localhost/db",
])
def test_placeholder_or_invalid_urls_are_not_configured(url):
    assert is_database_configured(url) is False


def test_real_looking_url_is_configured():
    url = "postgresql://postgres.abcd:secret@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
    assert is_database_configured(url) is True


def test_no_connection_attempt_with_placeholder(monkeypatch):
    monkeypatch.setattr(config, "SUPABASE_DATABASE_URL", "YOUR_SUPABASE_DATABASE_URL")

    def fail(*args, **kwargs):
        raise AssertionError("Tried to connect with a placeholder URL")
    monkeypatch.setattr(database.psycopg, "connect", fail)

    with pytest.raises(DatabaseNotConfiguredError):
        database.get_connection()


def test_password_is_not_shown():
    url = "postgresql://postgres.abcd:SuperSecret123@db.example.com:5432/postgres"
    shown = database.describe_connection_target(url)

    assert "SuperSecret123" not in shown
    assert "db.example.com" in shown


def test_sql_vector_size_matches_embedding_dimension():
    sql = SQL_FILE.read_text(encoding="utf-8")
    sizes = re.findall(r"^\s*embedding\s+(?:vector|halfvec)\((\d+)\)", sql, re.MULTILINE)

    assert sizes == [str(config.EMBEDDING_DIMENSION)], (
        "The vector size in the SQL file must match EMBEDDING_DIMENSION in .env"
    )


def test_pgvector_versions_compare_numerically():
    assert database.parse_version("0.8.2") >= database.MIN_PGVECTOR_VERSION
    assert database.parse_version("0.10.0") >= database.MIN_PGVECTOR_VERSION
    assert database.parse_version("0.7.4") < database.MIN_PGVECTOR_VERSION


@pytest.mark.integration
@pytest.mark.skipif(not is_database_configured(), reason="SUPABASE_DATABASE_URL not set in .env")
def test_real_database_is_ready():
    result = database.check_database()

    assert result["problems"] == []
