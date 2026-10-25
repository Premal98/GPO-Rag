# Testing guide: three document chatbots

This walks through testing everything end to end: the GPO report (`narrative`), the
Apple iMac environmental report (`visual`) and the World Bank report (`structured`).
Everything below uses the interactive API page at <http://127.0.0.1:8000/docs>; no
other tools are needed.

Time needed: about 15 minutes of your time, plus waiting for the World Bank
report to process (about 15-25 minutes in the background).

---

## 0. One-time setup

1. **Install the packages** (see [README](README.md#1-install) for details):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Create your `.env`** from the template and fill in `GOOGLE_API_KEY` and
   `SUPABASE_DATABASE_URL` (see [README](README.md#2-configure-env)):

   ```powershell
   Copy-Item .env.example .env         # macOS/Linux: cp .env.example .env
   ```

   All other settings have working defaults. `MAX_UPLOAD_MB=50` is enough for the
   three reports.

3. **Database:** in the Supabase SQL Editor, run `sql/001_create_document_chunks.sql`
   and then `sql/002_documents_and_chunk_metadata.sql`
   (see [README](README.md#3-set-up-supabase)). Then check the setup:

   ```powershell
   python -m scripts.check_database
   ```

4. **Unit tests** (no credentials, no quota, ~10 seconds):

   ```powershell
   python -m pytest
   ```

   Expect about 320 passed and a few skipped.

5. **Start the API:**

   ```powershell
   uvicorn app.main:app
   ```

   Use it **without** `--reload` while testing: a reload restarts the server and
   stops any document that is being processed.

   The first start downloads the reranker model (~1.1 GB) and loads it in the
   background. The first Docling run also downloads its models (a few hundred MB).
   Both happen only once.

---

## 1. GPO report (`narrative`)

1. `POST /documents`: choose the GPO annual report PDF, set `profile` to `narrative`.
   You get `202` and a `document_id`. Document IDs are derived from the file's
   contents, so the same PDF always gets `9e4f9e917d951597`; the examples below use
   that ID. If yours differs, use your own.
2. Poll `GET /documents/{document_id}` until `status` is `ready`. Narrative documents
   use fast plain-text parsing, so this takes about a minute (mostly embedding).

**Check:**

- `GET /documents`: GPO is listed with `profile: narrative`, `status: ready`.
- `POST /chat`:

  ```json
  {"question": "What was GPO's total revenue in FY 2018?", "document_id": "9e4f9e917d951597"}
  ```

  You should get `status: answered`, a figure with a citation like
  `(GPO-Anual-report.pdf, p. 6)`, and `retrieval.reranked: true`.

---

## 2. Apple iMac report (`visual`)

1. `POST /documents`: choose the PDF, set `profile` to `visual`. You get `202` and a
   `document_id`. Copy it.
2. Poll `GET /documents/{document_id}` every 30 seconds or so. `status` moves through
   `uploaded -> parsing -> enriching -> embedding -> ready`. For a ~20-30 page report
   expect roughly 2-5 minutes.

**Check the processing result** (`processing` in the same response):

- `parser: docling`
- `layout_summary.by_type` lists `table`, `picture`, probably `footnote`
- `warnings`: ideally empty. Gemini rate-limit warnings are OK (see section 6).

**Check the chunks:** `GET /documents/{document_id}/chunks`. Look for chunks whose
`metadata.chunk_type` is:

- `table`: starts with `Table: ...`, then `Summary: ...` (written by Gemini), then a
  markdown table, and possibly `Footnotes on this page:`
- `figure`: `Figure: <caption>`, a description of the chart, and
  `Data read from the chart:` with a small table
- `footnote`: `Footnote: ...`
- `text`: with a `section_path` such as `Climate Change > Carbon footprint`

**Chat** (`POST /chat` with this `document_id`). Good questions to try:

| Type | Question |
|---|---|
| table | What is the total carbon footprint of the iMac in kg CO2e? |
| chart | Which life cycle phase contributes most to the iMac's emissions? |
| footnote | Which baseline year are the emission reductions compared against? |
| fact | What percentage of the packaging is fiber-based? |

What to look for in the response:

- `sources[].chunk_type` shows `table` / `figure` for table and chart questions
- numbers are quoted exactly with their units
- footnote caveats are mentioned where relevant
- `retrieval.profile: visual`

---

## 3. World Bank report (`structured`)

1. `POST /documents` with the PDF, `profile` = `structured`. Copy the `document_id`.
2. Poll `GET /documents/{document_id}`. For ~200 pages expect about **15-25 minutes**:
   - Docling: about 8 minutes
   - Gemini enrichment: charts, tables and chapter summaries, spaced out for
     free-tier limits
   - embedding: about 1,000+ chunks at free-tier speed

   You can test the other bots while it runs. Asking this one a question before it's
   ready returns `409 still being processed`, which is expected.

**Check the chunks** (`GET /documents/{document_id}/chunks`):

- `chunk_id`s ending in `_p<n>` are **parent** sections (`metadata.embed: false`)
- `..._p<n>_c<m>` are their **children** (searched)
- `summary` chunks: `Summary of <chapter>:` and `Summary of the whole document:`
- `section_path` values look like `1 Global Outlook > 1.2 Inflation`

**Chat** questions to try:

| Type | Question |
|---|---|
| broad | What is the main message of the report? |
| section | What does chapter 2 conclude? |
| fact | What growth rate does the report project for the global economy? |
| table | Which region is projected to grow fastest? |

What to look for in the response:

- broad questions cite `summary` sources
- `What does chapter 2 conclude?` shows `retrieval.section_filter` set, and all
  sources have a `section_path` in chapter 2
- sources for detailed questions show a page range (`page_number` to `page_end`)
  because the whole parent section is used

---

## 4. Isolation between the bots

Ask the GPO bot an Apple question:

```json
{"question": "What is the carbon footprint of the iMac?", "document_id": "9e4f9e917d951597"}
```

It should answer `not_supported` (or `no_context`), and never cite the Apple PDF.
Repeat with the other combinations if you like.

---

## 5. Measure retrieval quality (optional, recommended for the demo)

1. Open `eval/apple_questions.template.json` and
   `eval/world_bank_questions.template.json`.
2. For each question, find the page(s) of the answer in the PDF and put them in
   `expected_pages`, e.g. `[12]`. Delete questions the report doesn't answer, and
   add your own.
3. Run:

   ```powershell
   python -m scripts.evaluate_retrieval eval/retrieval_questions.example.json
   python -m scripts.evaluate_retrieval eval/apple_questions.template.json --document-id <apple id>
   python -m scripts.evaluate_retrieval eval/world_bank_questions.template.json --document-id <world bank id>
   ```

   You get hit@k / recall@k / MRR overall and **per question type** (table, chart,
   footnote, section, broad). A hit@k of 0.8 or more is good. Look at the misses to
   see which kind of question needs work.

This only uses embedding quota (one request per question), not chat quota.

---

## 6. Things that may go wrong (and are handled)

| What you see | Meaning / what to do |
|---|---|
| `status: failed`, `status_message` explains | Read the message. Upload again (with `force=true` if needed). |
| `failed` with "The server restarted while..." | The server was restarted mid-job. Upload again. |
| `processing.warnings`: "could not be enriched" / "Enrichment stopped early" | Gemini free-tier limit. The document still works. Later, `POST /documents` again with `force=true` to finish only the missing items. |
| `POST /documents` returns 200 `accepted: false` | The document is already ready (or already queued). Add `force=true` to reprocess. |
| First chat after a server start is slow | The reranker is still loading (a few seconds). |
| A search response has the warning "Reranking was skipped" | The reranker failed to load; results are still returned. |

---

## 7. Clean up (optional)

`DELETE /documents/{document_id}` removes a document's chunks, its entry, its
uploaded PDF, cached processing results and chart images.
