"""Checks the Supabase database connection and setup.

Run from the project folder:
    python -m scripts.check_database
"""

import sys

import psycopg

from app import config
from app.database import (
    MIGRATION_002,
    MIN_PGVECTOR_VERSION,
    DatabaseNotConfiguredError,
    check_database,
    describe_connection_target,
    is_database_configured,
    parse_version,
)


def main():
    if not is_database_configured():
        print("[SKIP] SUPABASE_DATABASE_URL is not set (or is still a placeholder) in .env.")
        print("       Nothing was connected. See .env.example for the format.")
        return 1

    print(f"Connecting to {describe_connection_target()} ...")
    try:
        result = check_database()
    except DatabaseNotConfiguredError as err:
        print(f"[FAIL] {err}")
        return 1
    except psycopg.OperationalError as err:
        print("[FAIL] Could not connect to the database.")
        print(f"       {str(err).strip()}")
        print("       Check the password, host, and that you used the Session pooler string.")
        return 1

    print(f"[OK]   Connected. PostgreSQL {result['server_version']}")
    version = result["pgvector_version"]
    version_ok = version is not None and parse_version(version) >= MIN_PGVECTOR_VERSION
    print(f"[{'OK' if version_ok else 'FAIL'}]".ljust(7)
          + f"pgvector: {version or 'not enabled'} (0.8.0 or newer needed)")
    print(f"[{'OK' if result['table_exists'] else 'FAIL'}]".ljust(7)
          + f"document_chunks table: {'found' if result['table_exists'] else 'missing'}")
    if result["table_exists"]:
        ok = result["embedding_dimension"] == result["expected_dimension"]
        print(f"[{'OK' if ok else 'FAIL'}]".ljust(7)
              + f"embedding dimension: {result['embedding_dimension']} "
              + f"(expected {result['expected_dimension']} for {config.EMBEDDING_MODEL})")
        print(f"[{'OK' if result['migrated'] else 'FAIL'}]".ljust(7)
              + f"documents table and chunk metadata ({MIGRATION_002}): "
              + ("found" if result["migrated"] else "missing"))

    for problem in result["problems"]:
        print(f"  -> {problem}")
    if result["problems"]:
        return 1
    print("Database is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
