-- =====================================================================
-- Setup for the RAG chatbot: pgvector + document_chunks table
--
-- Run this once in Supabase: Dashboard > SQL Editor > New query > paste > Run.
-- It is safe to run more than once ("if not exists" everywhere).
--
-- EMBEDDING SIZE: vector(1536) must match EMBEDDING_DIMENSION in your .env.
--   Model: gemini-embedding-2 (default 3072 dims; Google recommends
--   768, 1536 or 3072). We use 1536 because pgvector's HNSW index supports
--   at most 2000 dimensions for the "vector" type.
--   To use a different size, change 1536 below AND EMBEDDING_DIMENSION
--   before creating the table. (For 3072, use halfvec(3072) and
--   halfvec_cosine_ops instead, which can be indexed up to 4000 dims.)
-- =====================================================================

-- 1. Enable pgvector (Supabase keeps extensions in the "extensions" schema)
create extension if not exists vector with schema extensions;

-- 2. One row per chunk produced by the chunking step
create table if not exists public.document_chunks (
    chunk_id        text primary key,                 -- e.g. 7b7555465a015cd7_p80_c1
    document_id     text not null,                    -- hash of the uploaded file
    content         text not null check (length(trim(content)) > 0),
    source_filename text not null,
    page_number     integer not null check (page_number > 0),
    chunk_index     integer not null check (chunk_index >= 0),
    metadata        jsonb not null default '{}'::jsonb,
    embedding       vector(1536),                     -- filled in by the ingestion step
    created_at      timestamptz not null default now()
);

-- 3. Fast lookups of all chunks for one document (e.g. to delete or re-ingest it)
create index if not exists document_chunks_document_id_idx
    on public.document_chunks (document_id);

-- 4. Fast similarity search on embeddings (cosine distance, operator <=>)
create index if not exists document_chunks_embedding_idx
    on public.document_chunks
    using hnsw (embedding vector_cosine_ops);

-- 5. Block access through Supabase's public API (anon/authenticated keys).
--    Our FastAPI backend connects as the database owner, so it is not affected.
alter table public.document_chunks enable row level security;
