"""Ingests a processed document: embeds its chunks and stores them in Supabase.

The PDF must have been uploaded first (POST /documents/upload), which
extracts and chunks it. This step only reads those saved chunks.
"""

import time
from pathlib import Path

import psycopg

from app import config
from app.database import MIGRATION_002, is_database_configured
from app.document_store import load_result
from app.embeddings import (
    EmbeddingError,
    create_embedder,
    embed_texts,
    format_document_for_embedding,
    is_embedding_configured,
    is_rate_limited,
)
from app.profiles import DEFAULT_PROFILE
from app.vector_store import ChunkStore


class IngestionError(Exception):
    """A problem we report back to the user, with an HTTP status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def ingest_document(document_id, upload_dir, force=False, embedder=None, store=None,
                    sleep=time.sleep, processed=None):
    """Embed a document's chunks and save them to the database.

    force=False: if the document is already in the database, do nothing.
    force=True:  replace the document's existing rows.

    embedder, store, and sleep can be replaced in tests.
    """
    # processed: an already loaded result (the background job passes its own)
    if processed is None:
        processed = load_result(upload_dir, document_id)
    if processed is None:
        raise IngestionError("Document not found. Upload the PDF first.", 404)
    chunks = processed["chunks"]
    if not chunks:
        raise IngestionError("This document has no text chunks to ingest. It may need OCR.", 422)

    # Check settings before doing anything that costs time or API quota
    if embedder is None:
        if not is_embedding_configured():
            raise IngestionError("GOOGLE_API_KEY is not set in .env.", 503)
        embedder = create_embedder()
    if store is None:
        if not is_database_configured():
            raise IngestionError("SUPABASE_DATABASE_URL is not set in .env.", 503)
        store = ChunkStore()

    try:
        setup = store.status()
        existing = store.count_chunks(document_id)
    except psycopg.Error as err:
        raise IngestionError(f"Database error: {str(err).strip()}", 503) from err

    column_dimension = setup["embedding_dimension"]
    if column_dimension is None:
        raise IngestionError(
            "The document_chunks table does not exist. Run sql/001_create_document_chunks.sql.", 503
        )
    if column_dimension != config.EMBEDDING_DIMENSION:
        raise IngestionError(
            f"The database embedding column has {column_dimension} dimensions, but "
            f"EMBEDDING_DIMENSION is {config.EMBEDDING_DIMENSION}. They must match.", 409
        )
    if not setup["migrated"]:
        raise IngestionError(f"The documents table is missing. Run {MIGRATION_002}.", 503)

    document = {
        "document_id": document_id,
        "filename": processed["filename"],
        "profile": processed.get("profile", DEFAULT_PROFILE),
        "page_count": processed["page_count"],
    }
    summary = {
        "document_id": document_id,
        "filename": processed["filename"],
        "profile": document["profile"],
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "chunk_count": len(chunks),
    }

    # Protection against inserting the same document twice
    if existing and not force:
        return {
            **summary,
            "status": "already_ingested",
            "message": (f"This document already has {existing} chunks in the database. "
                        "Nothing was changed. Use force=true to replace them."),
            "chunks_inserted": 0,
        }

    # Parent chunks (structured profile) are only read, never searched, so
    # they get no embedding. The section path in the title gives each
    # embedded chunk its context, e.g. "Report > 2 Outlook > 2.1 Inflation".
    stem = Path(processed["filename"]).stem
    embedded = [c for c in chunks if c["metadata"].get("embed", True)]
    texts = [
        format_document_for_embedding(
            c["text"],
            f"{stem} > {c['metadata']['section_path']}" if c["metadata"].get("section_path") else stem,
            config.EMBEDDING_MODEL,
        )
        for c in embedded
    ]
    try:
        vectors = embed_texts(
            embedder,
            texts,
            expected_dimension=config.EMBEDDING_DIMENSION,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            max_retries=config.EMBEDDING_MAX_RETRIES,
            sleep=sleep,
        )
    except EmbeddingError as err:
        if is_rate_limited(err):
            raise IngestionError(
                "The Gemini embedding quota was reached, so nothing was saved. Wait a minute "
                f"and try again, or enable billing for higher limits. Details: {err}", 429
            ) from err
        raise IngestionError(str(err), 502) from err

    vector_by_id = {c["chunk_id"]: vector for c, vector in zip(embedded, vectors)}
    rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": document_id,
            "content": chunk["text"],
            "source_filename": chunk["metadata"]["source"],
            "page_number": chunk["metadata"]["page"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "metadata": {**chunk["metadata"], "embedding_model": config.EMBEDDING_MODEL},
            "embedding": vector_by_id.get(chunk["chunk_id"]),
            # Set by the structured and visual chunkers; plain page chunks use the defaults
            "chunk_type": chunk["metadata"].get("chunk_type", "text"),
            "section_path": chunk["metadata"].get("section_path"),
            "page_end": chunk["metadata"].get("page_end", chunk["metadata"]["page"]),
            "parent_chunk_id": chunk["metadata"].get("parent_chunk_id"),
            "printed_page": chunk["metadata"].get("printed_page"),
        }
        for chunk in chunks
    ]

    try:
        saved = store.save_chunks(document, rows, replace=force)
    except psycopg.Error as err:
        raise IngestionError(f"Database error while saving: {str(err).strip()}", 503) from err

    if saved["skipped"]:
        # Another request ingested this document while we were embedding
        return {
            **summary,
            "status": "already_ingested",
            "message": "This document was ingested by another request. Nothing was changed.",
            "chunks_inserted": 0,
        }

    replaced = force and saved["existing"] > 0
    return {
        **summary,
        "status": "reingested" if replaced else "ingested",
        "message": f"Stored {saved['inserted']} chunks with embeddings.",
        "chunks_inserted": saved["inserted"],
    }
