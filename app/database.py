"""Connects to Supabase PostgreSQL.

Nothing here runs when the app starts. A connection is only opened when
get_connection() is called, and only if a real connection string is set.
"""

import psycopg

from app import config

TABLE_NAME = "public.document_chunks"
DOCUMENTS_TABLE = "public.documents"

# hnsw.iterative_scan, which keeps per-document searches complete, needs 0.8.0
MIN_PGVECTOR_VERSION = (0, 8, 0)
MIGRATION_002 = "sql/002_documents_and_chunk_metadata.sql"

# True when sql/002 has run: the documents table and the keyword-search column exist
MIGRATED_SQL = f"""
    select to_regclass('{DOCUMENTS_TABLE}') is not null
       and exists (select 1 from pg_attribute
                   where attrelid = to_regclass('{TABLE_NAME}')
                     and attname = 'content_tsv' and not attisdropped)
"""


def parse_version(version):
    """'0.8.2' -> (0, 8, 2). Non-numeric parts are ignored."""
    return tuple(int(part) for part in version.split(".") if part.isdigit())


class DatabaseNotConfiguredError(Exception):
    """SUPABASE_DATABASE_URL is missing or still a placeholder."""


def is_database_configured(url=None):
    """Return True only if the connection string looks real (not a placeholder)."""
    url = (config.SUPABASE_DATABASE_URL if url is None else url).strip()
    if not url or "YOUR_" in url:
        return False
    return url.startswith(("postgresql://", "postgres://"))


def get_connection():
    """Open a new database connection. Use it in a "with" block so it closes."""
    url = config.SUPABASE_DATABASE_URL.strip()
    if not is_database_configured(url):
        raise DatabaseNotConfiguredError(
            "SUPABASE_DATABASE_URL is not set. Copy your Supabase connection "
            "string into the .env file (see .env.example)."
        )

    options = {
        "connect_timeout": 10,
        # Supabase's poolers don't support prepared statements in transaction mode
        "prepare_threshold": None,
    }
    if "sslmode=" not in url:
        options["sslmode"] = "require"  # Supabase requires encrypted connections
    return psycopg.connect(url, **options)


def describe_connection_target(url=None):
    """Return host/port/database/user for display, never the password."""
    url = config.SUPABASE_DATABASE_URL if url is None else url
    try:
        info = psycopg.conninfo.conninfo_to_dict(url)
    except psycopg.ProgrammingError:
        return "(could not read the connection string)"
    return f"{info.get('user')}@{info.get('host')}:{info.get('port', 5432)}/{info.get('dbname')}"


def check_database():
    """Connect and check that pgvector, both SQL scripts and the tables are set up.

    Only reads from the database. Returns a dictionary describing what was found;
    "problems" is an empty list when everything is ready.
    """
    with get_connection() as conn:
        server_version = conn.execute("show server_version").fetchone()[0]

        row = conn.execute("select extversion from pg_extension where extname = 'vector'").fetchone()
        pgvector_version = row[0] if row else None

        table_exists = conn.execute("select to_regclass(%s) is not null", (TABLE_NAME,)).fetchone()[0]

        # For pgvector columns, atttypmod holds the number of dimensions
        row = conn.execute(
            """
            select atttypmod from pg_attribute
            where attrelid = to_regclass(%s) and attname = 'embedding' and not attisdropped
            """,
            (TABLE_NAME,),
        ).fetchone()
        embedding_dimension = row[0] if row else None

        migrated = conn.execute(MIGRATED_SQL).fetchone()[0]

    problems = []
    if pgvector_version is None:
        problems.append("The pgvector extension is not enabled. Run sql/001_create_document_chunks.sql.")
    elif parse_version(pgvector_version) < MIN_PGVECTOR_VERSION:
        problems.append(
            f"pgvector {pgvector_version} is too old; 0.8.0 or newer is needed. Upgrade it in "
            "Supabase under Database > Extensions > vector."
        )
    if not table_exists:
        problems.append(f"Table {TABLE_NAME} does not exist. Run sql/001_create_document_chunks.sql.")
    elif embedding_dimension != config.EMBEDDING_DIMENSION:
        problems.append(
            f"The embedding column has {embedding_dimension} dimensions but "
            f"EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}. They must match."
        )
    if table_exists and not migrated:
        problems.append(f"The documents table or new chunk columns are missing. Run {MIGRATION_002}.")

    return {
        "server_version": server_version,
        "pgvector_version": pgvector_version,
        "table_exists": table_exists,
        "embedding_dimension": embedding_dimension,
        "expected_dimension": config.EMBEDDING_DIMENSION,
        "migrated": migrated,
        "problems": problems,
    }
