"""Reads and writes documents and their chunks (with embeddings) in Supabase."""

from psycopg.types.json import Jsonb

from app.database import DOCUMENTS_TABLE, MIGRATED_SQL, TABLE_NAME, get_connection

INSERT_SQL = f"""
    insert into {TABLE_NAME}
        (chunk_id, document_id, content, source_filename, page_number,
         chunk_index, metadata, embedding, chunk_type, section_path,
         page_end, parent_chunk_id, printed_page)
    values (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s)
    on conflict (chunk_id) do nothing
"""

# Registers the document before its chunks are inserted (the chunks
# reference it), or updates it when the document is re-ingested
UPSERT_DOCUMENT_SQL = f"""
    insert into {DOCUMENTS_TABLE}
        (document_id, filename, profile, page_count, status, status_message, ingested_at)
    values (%(document_id)s, %(filename)s, %(profile)s, %(page_count)s, 'ready', null, now())
    on conflict (document_id) do update set
        filename = excluded.filename, profile = excluded.profile,
        page_count = excluded.page_count, status = 'ready', status_message = null,
        ingested_at = now()
"""

# Records progress of background processing. A profile of None keeps the
# stored one (new rows start as narrative until the profile is known).
SET_STATUS_SQL = f"""
    insert into {DOCUMENTS_TABLE} (document_id, filename, profile, status, status_message)
    values (%(document_id)s, %(filename)s, coalesce(%(profile)s, 'narrative'),
            %(status)s, %(message)s)
    on conflict (document_id) do update set
        filename = excluded.filename,
        profile = coalesce(%(profile)s, {DOCUMENTS_TABLE}.profile),
        status = excluded.status, status_message = excluded.status_message
"""

# Statuses of documents that are still being processed
BUSY_STATUSES = ("uploaded", "parsing", "enriching", "embedding")

DOCUMENT_COLUMNS = ("document_id", "filename", "profile", "status", "status_message",
                    "page_count", "chunk_count", "created_at", "ingested_at")
SELECT_DOCUMENTS_SQL = f"select {', '.join(DOCUMENT_COLUMNS)} from {DOCUMENTS_TABLE}"

SEARCH_COLUMNS = ("chunk_id", "document_id", "content", "source_filename", "page_number",
                  "chunk_index", "metadata", "chunk_type", "section_path", "page_end",
                  "printed_page", "parent_chunk_id")

# <=> is pgvector's cosine distance (0 = same direction, 2 = opposite). It
# matches the vector_cosine_ops HNSW index, so ordering by it uses the index.
SEARCH_SQL = f"""
    select {', '.join(SEARCH_COLUMNS)}, embedding <=> %(query)s::vector as distance
    from {TABLE_NAME}
    where embedding is not null {{document_filter}}
    order by embedding <=> %(query)s::vector
    limit %(limit)s
"""

# Keyword search on the content_tsv column (sql/002). terms is an OR query
# like 'gpo | revenue | 2018'; chunks matching more (and rarer) terms rank
# higher. The cosine distance is computed too, so every candidate has a
# similarity score.
KEYWORD_SQL = f"""
    select {', '.join(SEARCH_COLUMNS)}, embedding <=> %(query)s::vector as distance
    from {TABLE_NAME}
    where embedding is not null
      and content_tsv @@ to_tsquery('english', %(terms)s) {{filters}}
    order by ts_rank_cd(content_tsv, to_tsquery('english', %(terms)s)) desc
    limit %(limit)s
"""

# The HNSW index only looks at ef_search candidates (default 40), which caps
# how many rows a search can return. It is raised per query when needed.
DEFAULT_EF_SEARCH = 40

# Without this, the index finds the ef_search nearest chunks of ALL documents
# and only then applies the document filter, so a small document next to a
# big one can get few or no results. With it (pgvector 0.8+), the index keeps
# scanning until enough rows pass the filter. strict_order keeps the results
# sorted by distance.
ITERATIVE_SCAN = "strict_order"


def to_vector_literal(vector):
    """Format a list of numbers the way pgvector expects: '[0.1,0.2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class ChunkStore:
    """All database access for documents and chunks. Tests replace this with a fake."""

    def __init__(self, connect=get_connection):
        self._connect = connect

    def status(self, document_id=None):
        """Check the database setup, in a single connection.

        Returns {"embedding_dimension": size of the embedding column, or None
        if the table doesn't exist; "migrated": True once sql/002 has run;
        "document": the registered document for document_id, or None}.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                select atttypmod from pg_attribute
                where attrelid = to_regclass(%s) and attname = 'embedding' and not attisdropped
                """,
                (TABLE_NAME,),
            ).fetchone()
            dimension = row[0] if row else None
            migrated = dimension is not None and conn.execute(MIGRATED_SQL).fetchone()[0]

            document = None
            if migrated and document_id is not None:
                row = conn.execute(SELECT_DOCUMENTS_SQL + " where document_id = %s",
                                   (document_id,)).fetchone()
                document = dict(zip(DOCUMENT_COLUMNS, row)) if row else None

        return {"embedding_dimension": dimension, "migrated": bool(migrated), "document": document}

    def count_chunks(self, document_id):
        with self._connect() as conn:
            return conn.execute(
                f"select count(*) from {TABLE_NAME} where document_id = %s", (document_id,)
            ).fetchone()[0]

    def save_chunks(self, document, rows, replace=False):
        """Register the document and insert all its rows in a single transaction.

        document has document_id, filename, profile and page_count.
        - If the document already has rows and replace is False, nothing is
          changed and {"skipped": True} is returned.
        - If replace is True, the old rows are deleted first.
        - If anything fails, nothing is saved (the transaction is rolled back).
        """
        document_id = document["document_id"]
        with self._connect() as conn:
            with conn.transaction():
                # Stops two requests ingesting the same document at the same time
                conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (document_id,))

                existing = conn.execute(
                    f"select count(*) from {TABLE_NAME} where document_id = %s", (document_id,)
                ).fetchone()[0]
                if existing and not replace:
                    return {"skipped": True, "inserted": 0, "existing": existing}
                if replace:
                    conn.execute(f"delete from {TABLE_NAME} where document_id = %s", (document_id,))

                conn.execute(UPSERT_DOCUMENT_SQL, document)
                with conn.cursor() as cur:
                    cur.executemany(INSERT_SQL, [
                        (
                            row["chunk_id"],
                            row["document_id"],
                            row["content"],
                            row["source_filename"],
                            row["page_number"],
                            row["chunk_index"],
                            Jsonb(row["metadata"]),
                            # None for parent chunks, which are read but never searched
                            to_vector_literal(row["embedding"]) if row["embedding"] is not None else None,
                            row["chunk_type"],
                            row["section_path"],
                            row["page_end"],
                            row["parent_chunk_id"],
                            row["printed_page"],
                        )
                        for row in rows
                    ])
                    inserted = cur.rowcount
                conn.execute(f"update {DOCUMENTS_TABLE} set chunk_count = %s where document_id = %s",
                             (inserted, document_id))

        return {"skipped": False, "inserted": inserted, "existing": existing}

    def count_embedded_chunks(self, document_id=None):
        """Count rows that have an embedding, for all documents or just one."""
        sql = f"select count(*) from {TABLE_NAME} where embedding is not null"
        params = ()
        if document_id is not None:
            sql += " and document_id = %s"
            params = (document_id,)
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone()[0]

    @staticmethod
    def _filters(params, document_id, section_pattern):
        sql = ""
        if document_id is not None:
            sql += " and document_id = %(document_id)s"
            params["document_id"] = document_id
        if section_pattern is not None:
            sql += " and section_path ~* %(section)s"  # case-insensitive regex
            params["section"] = section_pattern
        return sql

    def _vector_rows(self, conn, query_vector, limit, document_id, section_pattern):
        params = {"query": to_vector_literal(query_vector), "limit": limit}
        sql = SEARCH_SQL.format(document_filter=self._filters(params, document_id, section_pattern))
        # "true" makes the settings last only for the current transaction
        conn.execute(
            "select set_config('hnsw.ef_search', %s, true), "
            "set_config('hnsw.iterative_scan', %s, true)",
            (str(max(DEFAULT_EF_SEARCH, limit)), ITERATIVE_SCAN),
        )
        return [dict(zip(SEARCH_COLUMNS + ("distance",), row))
                for row in conn.execute(sql, params).fetchall()]

    def _keyword_rows(self, conn, terms, query_vector, limit, document_id, section_pattern):
        if not terms:
            return []
        params = {"query": to_vector_literal(query_vector), "terms": " | ".join(terms),
                  "limit": limit}
        sql = KEYWORD_SQL.format(filters=self._filters(params, document_id, section_pattern))
        return [dict(zip(SEARCH_COLUMNS + ("distance",), row))
                for row in conn.execute(sql, params).fetchall()]

    def search(self, query_vector, limit, document_id=None, section_pattern=None):
        """Return the `limit` chunks closest to query_vector, nearest first.

        Each result is a dict with the chunk's columns plus "distance"
        (cosine distance). Pass document_id to search a single document, and
        section_pattern (a regex) to search only matching sections.
        """
        with self._connect() as conn:
            with conn.transaction():
                return self._vector_rows(conn, query_vector, limit, document_id, section_pattern)

    def search_candidates(self, query_vector, terms, limit, document_id=None, section_pattern=None):
        """Vector and keyword search in one connection: (vector_rows, keyword_rows).

        terms are plain words (letters and digits only) for the keyword search.
        """
        with self._connect() as conn:
            with conn.transaction():
                vector = self._vector_rows(conn, query_vector, limit, document_id, section_pattern)
                keyword = self._keyword_rows(conn, terms, query_vector, limit, document_id,
                                             section_pattern)
        return vector, keyword

    def get_chunks(self, chunk_ids):
        """Chunks by id (e.g. the parents of matched chunks), as {chunk_id: row}."""
        if not chunk_ids:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"select {', '.join(SEARCH_COLUMNS)} from {TABLE_NAME} where chunk_id = any(%s)",
                (list(chunk_ids),),
            ).fetchall()
        return {row[0]: dict(zip(SEARCH_COLUMNS, row)) for row in rows}

    def get_footnotes(self, document_id, pages):
        """Footnote chunks on the given pages, as {page: [text, ...]}."""
        if not pages:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                f"select page_number, content from {TABLE_NAME} "
                f"where document_id = %s and chunk_type = 'footnote' and page_number = any(%s) "
                f"order by chunk_index",
                (document_id, sorted(pages)),
            ).fetchall()
        footnotes = {}
        for page, content in rows:
            footnotes.setdefault(page, []).append(content)
        return footnotes

    def set_document_status(self, document_id, filename, status, message=None, profile=None):
        """Create or update the document's registry row with a processing status."""
        with self._connect() as conn:
            conn.execute(SET_STATUS_SQL, {"document_id": document_id, "filename": filename,
                                          "profile": profile, "status": status, "message": message})

    def fail_unfinished(self, message):
        """Mark documents left mid-processing (e.g. by a server restart) as failed."""
        with self._connect() as conn:
            return conn.execute(
                f"update {DOCUMENTS_TABLE} set status = 'failed', status_message = %s "
                f"where status = any(%s)",
                (message, list(BUSY_STATUSES)),
            ).rowcount

    def list_documents(self):
        """All registered documents, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(SELECT_DOCUMENTS_SQL + " order by created_at, filename").fetchall()
        return [dict(zip(DOCUMENT_COLUMNS, row)) for row in rows]

    def delete_document(self, document_id):
        """Remove a document and all its chunks.

        Returns {"document_deleted": bool, "chunks_deleted": int}.
        """
        with self._connect() as conn:
            with conn.transaction():
                chunks = conn.execute(
                    f"delete from {TABLE_NAME} where document_id = %s", (document_id,)
                ).rowcount
                documents = conn.execute(
                    f"delete from {DOCUMENTS_TABLE} where document_id = %s", (document_id,)
                ).rowcount
        return {"document_deleted": documents > 0, "chunks_deleted": chunks}
