# GPO RAG Chatbot

A FastAPI backend that answers questions about your PDF files. Upload a PDF, store
its text as embeddings in Supabase (PostgreSQL + pgvector), then ask questions.
Answers come from Google Gemini and cite the source filename and page numbers.

> **Testing the three chatbots end to end:** see [TESTING.md](TESTING.md).
>
> **Status:** all automated tests use mocks and run without credentials. The
> real Gemini and Supabase connections are covered by the integration tests,
> which only run once you add real credentials (see [Testing](#testing)).

## How it works

```
POST /documents                  upload a PDF; parse, enrich, chunk and embed it in the background
GET  /documents/{id}             its status: uploaded -> parsing -> enriching -> embedding -> ready
GET  /documents                  all documents: one chatbot per document_id
POST /chat                       question + document_id -> hybrid search in that document -> Gemini answers
POST /search                     see what /chat would retrieve, without generating an answer
DELETE /documents/{id}           remove a document's chunks, registry entry and files

Step by step (synchronous, for inspecting each stage):
POST /documents/upload           PDF -> extract/parse -> chunk (no database needed)
GET  /documents/{id}/chunks      look at the chunks
POST /documents/{id}/ingest      chunks -> Gemini embeddings -> Supabase
```

Each document is its own chatbot: `/chat` only searches the `document_id` you
pass, so answers never mix content from different documents. A frontend lists
`GET /documents` and offers one chat per document.

### Profiles

Every document has a profile that decides how it is processed and searched.
Choose one at upload with `?profile=...`, or leave it out and one is suggested
(`profile_reason` says why: 100+ pages -> structured, 12%+ numeric words -> visual).

| | `narrative` | `structured` | `visual` |
|---|---|---|---|
| Example | GPO annual report | World Bank report (~200 pages) | Apple iMac environmental report |
| Parsing | pypdf plain text (fast) | Docling layout analysis | Docling layout analysis |
| Text chunks | 1,000 characters per page | section-aware **parent/child**: 600-character children are searched, their ~3,000-character section is what the model reads | section-aware, 1,000 characters |
| Tables | in the page text | own chunks (markdown + Gemini summary) | own chunks (markdown + Gemini summary + page footnotes) |
| Charts | – | Gemini reads them (description + data table) | Gemini reads them (description + data table) |
| Footnotes | in the page text | in the section text | own chunks, attached to tables/figures/text on their page |
| Summaries | – | per chapter + whole document (for broad questions) | – |
| Default `top_k` | 5 | 6 | 8 |
| Extra prompt | – | mention sections, use summaries for broad questions | quote numbers with units, mention footnote caveats |

**Docling** ([docling-project/docling](https://github.com/docling-project/docling), MIT,
runs locally) finds headings with their levels, tables, figures (classified as
bar chart, photo, logo, ...), captions and footnotes, and drops running headers,
footers, contents and index pages. Scanned PDFs of any profile go through its OCR.

- It runs at roughly **2–3 seconds per page** on a CPU (82 pages in 190 s on 8 cores),
  so a 200-page report takes about 8 minutes, plus Gemini enrichment. This runs in
  the background and is cached in `uploads/processed/`; chart images are saved in
  `uploads/figures/`.
- The first run downloads its models (a few hundred MB) from Hugging Face.

**Enrichment** sends only charts and diagrams to Gemini (photos and logos are
skipped), several per request, spaced out for free-tier rate limits and capped by
`ENRICHMENT_MAX_*`. If a request fails, the document still works without that
description, and processing it again with `force=true` finishes the rest.

### Search

`/chat` and `/search` use **hybrid search**: vector similarity (pgvector) and
keyword search (Postgres full-text) each find 30 candidates; they are merged
(reciprocal rank fusion), the best 20 are **reranked** by a local cross-encoder
(BAAI/bge-reranker-base, MIT, ~1.1 GB download, ~2 s per question on a CPU), and
the top `top_k` are kept, up to 12,000 characters. A question that names a section
("what does chapter 3 say") searches only that section.

| Folder | Contents |
|---|---|
| `app/` | The API (`main.py`) and its services: PDF reading (`pdf_ingestion`, `parsing`), profiles, chunking (`chunking`, `chunkers`), Gemini enrichment, background jobs, embeddings, database, hybrid search and reranking, chat |
| `sql/` | Supabase setup scripts, run once each in order (`001_...`, `002_...`) |
| `scripts/` | `check_database`, `check_gemini`, `evaluate_retrieval` |
| `tests/` | Unit tests (mocked) and integration tests (real services, skipped by default) |
| `eval/` | Questions for measuring retrieval quality: GPO example, Apple and World Bank templates |
| `uploads/` | Uploaded PDFs and their extracted chunks (created at runtime, never committed) |

## Requirements

- Python 3.11+ (tested with 3.13)
- A Google Gemini API key
- A Supabase project (the free plan is enough)

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure `.env`

```powershell
Copy-Item .env.example .env         # macOS/Linux: cp .env.example .env
```

Fill in the two secrets. While a value still contains `YOUR_`, the app treats it
as missing and never tries to use it.

```ini
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
SUPABASE_DATABASE_URL=postgresql://postgres.YOUR_PROJECT_REF:YOUR_DB_PASSWORD@YOUR_POOLER_HOST:5432/postgres
```

- **`GOOGLE_API_KEY`:** create one at <https://aistudio.google.com/apikey>.
- **`SUPABASE_DATABASE_URL`:** in your Supabase project click **Connect** (top of
  the dashboard), choose **Session pooler**, copy the URI and put your database
  password in place of `[YOUR-PASSWORD]`. See [Supabase setup](#3-set-up-supabase).

`.env` is listed in `.gitignore` and must never be committed. All other settings
have working defaults and are explained in `.env.example`. The most important ones:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Changing it means re-ingesting every document (`force=true`). |
| `EMBEDDING_DIMENSION` | `1536` | Must equal `vector(1536)` in the SQL file. |
| `CHAT_MODEL` | `gemini-3.5-flash` | Google retires models; check the [deprecations page](https://ai.google.dev/gemini-api/docs/deprecations). |
| `RETRIEVAL_TOP_K` | `5` | Chunks per question when searching all documents (each profile has its own default). |
| `MAX_UPLOAD_MB` | `50` | Largest PDF accepted. |
| `RERANK_ENABLED` | `true` | Local reranker; `false` makes questions ~2 s faster but slightly less accurate. |
| `ENRICHMENT_ENABLED` | `true` | Gemini reads charts and summarizes tables/chapters of structured and visual documents. |
| `ENRICHMENT_MAX_FIGURES` / `_TABLES` / `_SECTIONS` | `40` / `80` / `30` | Caps on Gemini requests per document. |
| `ENRICHMENT_MIN_INTERVAL_SECONDS` | `4` | Gap between Gemini requests (free tier: about 15 per minute). |

## 3. Set up Supabase

1. Create a project at <https://supabase.com/dashboard>. Save the **database
   password** you choose; you need it for the connection string.
2. Open **SQL Editor** > **New query**, paste all of `sql/001_create_document_chunks.sql`
   and click **Run**. This:
   - enables the **pgvector** extension (you don't need to install anything else);
   - creates the `public.document_chunks` table, with `embedding vector(1536)`;
   - adds an HNSW index for cosine similarity search;
   - turns on row-level security, so Supabase's public API keys can't read the table.
     This backend connects as the database owner, so it is not affected.

   The script is safe to run more than once. You do not need to create a separate
   database: use the default `postgres` database that every Supabase project has.
3. In a new query, run all of `sql/002_documents_and_chunk_metadata.sql` the same way.
   This adds the `public.documents` table (what `GET /documents` lists), extra chunk
   columns (chunk type, section, page range) and a keyword-search index. It keeps
   existing chunks and registers already-ingested documents as `narrative`. It is
   also safe to run more than once. pgvector **0.8.0 or newer** is needed; check with
   `select extversion from pg_extension where extname = 'vector';` and upgrade under
   **Database > Extensions** if needed.
4. Put the Session pooler connection string in `.env` (step 2 above).
5. Check the setup:

   ```powershell
   python -m scripts.check_database
   ```

   It should report pgvector, the table, `embedding dimension: 1536 (expected 1536)`
   and the documents table.

**Updating an existing database:** future changes come as new numbered files in
`sql/`. Run each one once, in order, before starting the new code. After running
002, re-ingest existing documents with `?force=true` so they get the new metadata.

**Changing the embedding size:** edit `vector(1536)` in the SQL file and
`EMBEDDING_DIMENSION` together, *before* creating the table. pgvector's HNSW index
supports up to 2,000 dimensions for `vector`; for 3,072 use `halfvec(3072)` and
`halfvec_cosine_ops`. If the table already exists, drop it first
(`drop table public.document_chunks;`), which deletes all stored chunks.

## 4. Check Gemini

```powershell
python -m scripts.check_gemini
```

This makes one embedding call and one chat call, and does not touch the database.

## 5. Run the API

```powershell
uvicorn app.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>.

## API usage

You can also try every endpoint from the browser at <http://127.0.0.1:8000/docs>.
`curl.exe` is built into Windows 10+; on macOS/Linux use `curl`.

**Health check**

```powershell
curl.exe http://127.0.0.1:8000/health
```

**Upload and process a PDF (recommended).** Returns `202` at once with the
`document_id`; the work continues in the background. `profile` is optional.

```powershell
curl.exe -F "file=@C:\path\to\Apple-iMac.pdf;type=application/pdf" "http://127.0.0.1:8000/documents?profile=visual"
```

Then poll its status until `status` is `ready` (or `failed`, with `status_message`).
`processing` shows what was found: parser, element counts, warnings.

```powershell
curl.exe http://127.0.0.1:8000/documents/<document_id>
```

Uploading a file that is already `ready` returns `200` and does nothing, unless you
add `force=true` or pass a different profile. Jobs run one at a time, in order. A
server restart stops running jobs; their documents are marked `failed` and can be
uploaded again. While a document is processing, `/chat` returns `409`.

The steps below do the same thing synchronously, one stage at a time.

**1. Upload a PDF.** Returns a `document_id`. Only real, unencrypted `.pdf` files
up to `MAX_UPLOAD_MB` are accepted. Scanned (image-only) PDFs return 422 with
`status: "ocr_required"`, because no text can be extracted from them.

```powershell
curl.exe -F "file=@C:\path\to\GPO-Anual-report.pdf;type=application/pdf" http://127.0.0.1:8000/documents/upload
# With a profile: .../documents/upload?profile=visual   (narrative | structured | visual)
```

Uploading the same file again with a different profile re-chunks it; then ingest
with `?force=true` to replace the stored chunks.

**2. View the chunks** (optional, `?page=6` for one page)

```powershell
curl.exe http://127.0.0.1:8000/documents/9e4f9e917d951597/chunks
```

**3. Ingest:** embed the chunks and store them in Supabase. Running it again does
nothing unless you add `?force=true`, which replaces the stored chunks.

```powershell
curl.exe -X POST http://127.0.0.1:8000/documents/9e4f9e917d951597/ingest
```

**List the documents** (the chatbots). Each entry has `document_id`, `filename`,
`profile`, `status`, `page_count` and `chunk_count`.

```powershell
curl.exe http://127.0.0.1:8000/documents
```

**Delete a document:** `curl.exe -X DELETE http://127.0.0.1:8000/documents/9e4f9e917d951597`

**4. Search:** see which chunks a question retrieves, without generating an answer

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType "application/json" `
  -Body '{"query": "What was GPO revenue in FY 2018?", "top_k": 5}' | ConvertTo-Json -Depth 5
```

```bash
# macOS/Linux
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
  -d '{"query": "What was GPO revenue in FY 2018?", "top_k": 5}'
```

Each result has `content`, `filename`, `page_number`/`page_end`, `chunk_type`,
`section_path`, `similarity` (cosine, 1 = same meaning) and `rerank_score`. The
response also shows `candidates`, `reranked` and `section_filter`. Optional fields:
`document_id` (search one document), `top_k` and `min_similarity`.

**5. Chat.** `document_id` is required; an unknown one returns 404.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" `
  -Body '{"question": "What was GPO revenue in FY 2018?", "document_id": "9e4f9e917d951597"}' | ConvertTo-Json -Depth 5
```

```bash
# macOS/Linux
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"question": "What was GPO revenue in FY 2018?", "document_id": "9e4f9e917d951597"}'
```

Response, shortened:

```json
{
  "status": "answered",
  "answer": "Revenue totaled $874.5 million (GPO-Anual-report.pdf, p. 6).",
  "supported": true,
  "sources": [{"source_id": "S1", "filename": "GPO-Anual-report.pdf", "page_number": 6,
               "chunk_id": "9e4f9e917d951597_p6_c1", "similarity": 0.81, "excerpt": "..."}],
  "citations": [{"filename": "GPO-Anual-report.pdf", "pages": [6]}],
  "retrieval": {"status": "ok", "chunks_retrieved": 5, "top_k": 5},
  "warnings": [],
  "timing_ms": {"retrieval": 420.0, "generation": 1800.0}
}
```

(The numbers above show the response shape; they are not output from a real run.)

`status` is `answered`, `not_supported` (the documents don't contain the answer)
or `no_context` (nothing relevant was found, so Gemini was not called). Optional
request fields: `top_k`, `min_similarity`. Each source also has `chunk_type`
(`text`, `table`, `figure`, ...) and `section_path` when known.

### Error codes

| Code | Meaning |
|---|---|
| 400 | Not a valid PDF (wrong type, corrupted, password-protected, empty) |
| 404 | Unknown `document_id` |
| 409 | Document still processing or its processing failed (`/chat`, `/search`), or database column size doesn't match `EMBEDDING_DIMENSION` |
| 413 | PDF larger than `MAX_UPLOAD_MB` |
| 422 | Invalid request, or a scanned PDF with no text |
| 429 | Gemini rate limit reached |
| 502 | Gemini error (bad key, unknown model, unusable reply) |
| 503 | Credentials missing, table missing, or database unreachable |
| 504 | Gemini did not respond within the timeout |

## Testing

```powershell
python -m pytest
```

Unit tests replace Gemini and Supabase with fakes and mocks, so they need no
credentials and cost nothing. They always use placeholder credentials, even if
your `.env` has real ones (see `tests/conftest.py`).

The **integration tests** use the real services and are skipped until configured:

| Test | Needs |
|---|---|
| `tests/test_database.py::test_real_database_is_ready` | `SUPABASE_DATABASE_URL` |
| `tests/test_integration.py` (ingest, search, chat, unsupported question) | both secrets and `RUN_INTEGRATION_TESTS=1` |

```powershell
$env:RUN_INTEGRATION_TESTS = "1"; python -m pytest -m integration -v -s
Remove-Item Env:RUN_INTEGRATION_TESTS
```

They upload a tiny generated PDF, then parse it with real Docling, embed, store,
search (with the real reranker) and chat about it, and delete their rows afterwards.
Each run uses a few API calls of quota. `tests/test_parsing.py` also has a real
Docling test that needs no credentials.

### Evaluating retrieval quality

```powershell
python -m scripts.evaluate_retrieval eval/retrieval_questions.example.json
python -m scripts.evaluate_retrieval eval/apple_questions.template.json --document-id <id>
python -m scripts.evaluate_retrieval eval/world_bank_questions.template.json --document-id <id>
```

This reports hit@k, recall@k and MRR for questions whose answer pages you know, in
total and per question type (fact, table, chart, footnote, section, broad). A chunk
covering pages 4-6 counts as a hit for an answer on page 5. The two templates hold
typical questions for each report with empty `expected_pages`: read the PDF, fill in
the page(s) of each answer, and remove questions the report doesn't answer.
`--top-k` overrides the profile's default.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY is not set` / `SUPABASE_DATABASE_URL is not set` (503) | The value is empty or still contains `YOUR_`. Edit `.env` and restart uvicorn. |
| `could not translate host name` or a connection timeout | You are probably using the **Direct connection** string, which needs IPv6. Use the **Session pooler** string. |
| `password authentication failed` | Wrong database password. Reset it under Project Settings > Database. The pooler username is `postgres.<project-ref>`, not just `postgres`. |
| Password contains `@`, `:`, `/`, `#` or `%` | URL-encode those characters in the connection string (`@` becomes `%40`, `#` becomes `%23`), or choose a password without them. |
| `prepared statement ... already exists` | Happens with the transaction pooler (port 6543). This app already turns off prepared statements; port 5432 (session pooler) is recommended. |
| `The pgvector extension is not enabled` / `table ... does not exist` | Run `sql/001_create_document_chunks.sql` in the SQL Editor. |
| `The documents table is missing` (503) | Run `sql/002_documents_and_chunk_metadata.sql` in the SQL Editor. |
| `/chat` returns 404 for a document you uploaded | It isn't ingested yet. Call `POST /documents/{id}/ingest`, then check `GET /documents`. |
| `type "vector" does not exist` | pgvector was installed in a schema that isn't on the search path. Run `alter database postgres set search_path to "$user", public, extensions;`, then reconnect. |
| Dimension mismatch (409) | `EMBEDDING_DIMENSION` and `vector(N)` in the table differ. Make them match (see Supabase setup). |
| `The chat model ... was not found` (502) | `CHAT_MODEL` was retired or isn't available to your key. Pick a current model from the [models page](https://ai.google.dev/gemini-api/docs/models). |
| `/chat` returns 503 "model is overloaded" | Temporary high demand at Google. Retry in a minute, or set `CHAT_FALLBACK_MODEL` in `.env` so a second model answers when the main one is busy. |
| Rate limit (429) | Free-tier quotas are low. Wait, lower `EMBEDDING_BATCH_SIZE`, or enable billing. |
| `/chat` returns `no_context` | Nothing is ingested yet: upload, then call `/ingest`. Or `min_similarity` is set too high. |
| Warning that chunks were embedded with a different model | You changed `EMBEDDING_MODEL`. Re-ingest each document with `?force=true`. |
| Re-uploaded with new `CHUNK_SIZE` but the database still has the old chunks | Call `/ingest?force=true` for that document. |
| `ocr_required` | The PDF is scanned images and Docling's OCR found no text either. Check the file opens and isn't blank, or run it through another OCR tool (for example, `ocrmypdf`) and upload the result. |
| Upload of a structured/visual PDF takes minutes | Expected: Docling analyses every page on the CPU (about 2–3 s per page). It happens once per file. |
| `To support symlinks on Windows...` printed by Docling | Harmless Hugging Face warning; models are copied instead of linked. Turn on Windows Developer Mode to hide it. |
| Upload response has a warning that plain text extraction was used | Docling failed on this PDF (the warning says why), so pypdf's text was used. The document still works, without layout features. |
| `/chat` returns 409 "still being processed" | Wait until `GET /documents/{id}` shows `ready`. A large structured document can take 10-20 minutes (Docling, Gemini enrichment, embedding at free-tier rate limits). |
| Document stuck in `failed` with "The server restarted..." | Jobs stop when uvicorn restarts (including `--reload` after a code change). Upload the file again. |
| `processing.warnings` mentions figures/tables that could not be enriched, or "Enrichment stopped early" | Gemini rate limit or daily quota. The document works without those descriptions; upload it again later with `force=true` to finish (finished items are not sent again). |
| First question after starting the server takes much longer | The reranker model is loading (or downloading, ~1.1 GB, the very first time). It loads in the background at startup. |
| Search response warns "Reranking was skipped" | The reranker model could not be loaded (e.g. no internet for the first download). Results are still returned in hybrid order. Set `RERANK_ENABLED=false` to turn it off. |

## Security notes

- `.env` (and any `.env.*` except `.env.example`), `uploads/` and all `*.pdf` files
  are in `.gitignore`.
- The connection string gives full database access. Keep it on the server only.
- Uploads are limited to `MAX_UPLOAD_MB`, but the whole request body is received
  before that check. In production, also set a body-size limit on your reverse proxy.
- The API has no authentication. Don't expose it to the internet as it is.
