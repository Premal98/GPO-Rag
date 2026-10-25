-- =====================================================================
-- Step 2 of the database setup: document registry + richer chunk metadata
--
-- Run this once in Supabase, after 001_create_document_chunks.sql:
--   Dashboard > SQL Editor > New query > paste > Run.
-- It is safe to run more than once, and it keeps all existing chunks.
-- =====================================================================

-- 1. One row per ingested document. GET /documents lists this table.
create table if not exists public.documents (
    document_id     text primary key,                 -- hash of the uploaded file
    filename        text not null,
    profile         text not null default 'narrative'
                    check (profile in ('narrative', 'structured', 'visual')),
    status          text not null default 'ready'
                    check (status in ('uploaded', 'parsing', 'enriching', 'embedding', 'ready', 'failed')),
    status_message  text,
    page_count      integer check (page_count > 0),
    chunk_count     integer not null default 0 check (chunk_count >= 0),
    created_at      timestamptz not null default now(),
    ingested_at     timestamptz
);

alter table public.documents enable row level security;

-- 2. New chunk columns. Existing chunks get the defaults (plain text chunks).
alter table public.document_chunks
    add column if not exists chunk_type text not null default 'text'
        check (chunk_type in ('text', 'table', 'figure', 'footnote', 'summary')),
    add column if not exists section_path    text,       -- e.g. '3 Inflation > 3.2 Food prices'
    add column if not exists page_end        integer,    -- last page, for chunks that span pages
    add column if not exists parent_chunk_id text,       -- larger chunk this one belongs to
    add column if not exists printed_page    text,       -- page label printed on the page, e.g. 'xii'
    -- Keyword search (hybrid search), kept up to date by Postgres automatically
    add column if not exists content_tsv tsvector
        generated always as (to_tsvector('english', content)) stored;

create index if not exists document_chunks_content_tsv_idx
    on public.document_chunks using gin (content_tsv);

-- 3. Register documents that were ingested before this script existed
insert into public.documents (document_id, filename, profile, status, chunk_count, ingested_at)
select document_id, min(source_filename), 'narrative', 'ready', count(*), max(created_at)
from public.document_chunks
group by document_id
on conflict (document_id) do nothing;

-- 4. Deleting a document deletes its chunks too
do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'document_chunks_document_id_fkey') then
        alter table public.document_chunks
            add constraint document_chunks_document_id_fkey
            foreign key (document_id) references public.documents (document_id) on delete cascade;
    end if;
end $$;
