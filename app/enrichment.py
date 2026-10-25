"""Uses Gemini to add what text extraction can't see (structured and visual profiles).

  - figures: each chart or diagram image is described, and the data points
    Gemini can read are written out as a markdown table;
  - tables: a one-to-three sentence summary, embedded with the table so
    questions phrased in words find tables of numbers;
  - sections (structured): a summary per chapter and one for the document,
    which answer broad questions that no single chunk can.

Results are stored on the layout (element["description"], ["data"],
["summary"], layout["section_summaries"], layout["document_summary"]) and
saved with the processed document, so items done once are never sent again.
A failed batch only adds a warning; the document still works without it.
"""

import base64
import json
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app import config
from app.chunkers import build_sections, chapters
from app.embeddings import is_embedding_configured, is_rate_limited, suggested_retry_delay
from app.profiles import profile_setting

FIGURES_PER_REQUEST = 4
TABLE_CHARS_PER_REQUEST = 12_000
TABLE_MAX_CHARS_IN_PROMPT = 6_000
CHAPTER_MAX_CHARS_IN_PROMPT = 40_000
MAX_ATTEMPTS = 3
DEFAULT_RATE_LIMIT_WAIT = 30

FIGURE_PROMPT = """\
You are reading figures from the PDF document "{filename}". For each image below \
(F1, F2, ...), write:
- "description": 2-4 sentences: what kind of chart or diagram it is, what it shows, \
the axes and units, the time period, and the main trend or takeaway.
- "data": a markdown table of the data points you can read with confidence (labels, \
values with units, years). Use "" if no values can be read. Never guess a value.
Use only what is visible in the image and its caption.

{items}

Reply with JSON only: {{"figures": [{{"id": "F1", "description": "...", "data": "..."}}]}}"""

TABLE_PROMPT = """\
Below are tables (T1, T2, ...) from the PDF document "{filename}", as markdown. For \
each, write a "summary" of 1-3 sentences: what the table shows, its units and time \
period, and its most notable values. Use only the table's content.

{items}

Reply with JSON only: {{"tables": [{{"id": "T1", "summary": "..."}}]}}"""

CHAPTER_PROMPT = """\
Summarize this part of the PDF document "{filename}" in 5-8 sentences: its main \
points, conclusions and the most important numbers. Use only the text below.

Part: {title}

{text}

Reply with JSON only: {{"summary": "..."}}"""

DOCUMENT_PROMPT = """\
Below are summaries of each part of the PDF document "{filename}". Write a summary of \
the whole document in 6-10 sentences: its purpose, main themes and key conclusions. \
Use only these summaries.

{text}

Reply with JSON only: {{"summary": "..."}}"""


class EnrichmentError(Exception):
    """A Gemini request failed or returned something unusable."""


class StopEnrichment(EnrichmentError):
    """A limit was reached (e.g. the daily quota) that makes further requests pointless."""


def create_enrichment_model():
    return ChatGoogleGenerativeAI(
        model=config.ENRICHMENT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        timeout=config.ENRICHMENT_TIMEOUT_SECONDS,
        max_retries=0,  # retried below, honoring Gemini's suggested wait
        max_output_tokens=8192,
        response_mime_type="application/json",
    )


def parse_json(text):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise EnrichmentError("Gemini's reply was not valid JSON.") from err
    if not isinstance(data, dict):
        raise EnrichmentError("Gemini's reply was not a JSON object.")
    return data


class GeminiCaller:
    """Sends requests one at a time, spaced out, retrying after rate limits."""

    def __init__(self, llm, sleep=time.sleep, clock=time.monotonic):
        self.llm = llm
        self.sleep = sleep
        self.clock = clock
        self.last_call = None
        self.requests = 0

    def __call__(self, content):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.last_call is not None:
                wait = config.ENRICHMENT_MIN_INTERVAL_SECONDS - (self.clock() - self.last_call)
                if wait > 0:
                    self.sleep(wait)
            self.last_call = self.clock()
            self.requests += 1
            try:
                message = self.llm.invoke([HumanMessage(content=content)])
            except Exception as err:
                if "PerDay" in str(err):
                    raise StopEnrichment(f"The daily Gemini quota is used up ({err}).") from err
                if is_rate_limited(err) and attempt < MAX_ATTEMPTS:
                    self.sleep((suggested_retry_delay(err) or DEFAULT_RATE_LIMIT_WAIT) + 1)
                    continue
                raise EnrichmentError(f"Gemini request failed: {err}") from err
            text = message.content if isinstance(message.content, str) else "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in message.content)
            return parse_json(text)
        raise EnrichmentError("Gemini kept refusing requests because of rate limits.")


def _image_block(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _table_batches(tables):
    batch, size = [], 0
    for table in tables:
        length = min(len(table["text"]), TABLE_MAX_CHARS_IN_PROMPT)
        if batch and size + length > TABLE_CHARS_PER_REQUEST:
            yield batch
            batch, size = [], 0
        batch.append(table)
        size += length
    if batch:
        yield batch


def _call_batch(call, content, what, warnings):
    """Send one request; on failure add a warning and return None.

    StopEnrichment is raised, because later requests would fail too.
    """
    try:
        return call(content)
    except StopEnrichment:
        raise
    except EnrichmentError as err:
        warnings.append(f"Could not {what}: {err}")
        return None


def describe_figures(figures, call, filename, warnings):
    for batch in _batches(figures, FIGURES_PER_REQUEST):
        items, content = [], []
        for number, figure in enumerate(batch, start=1):
            caption = figure.get("caption") or "(no caption)"
            items.append(f"F{number}: page {figure['page']}, caption: {caption}")
        content.append({"type": "text", "text": FIGURE_PROMPT.format(
            filename=filename, items="\n".join(items))})
        for number, figure in enumerate(batch, start=1):
            content.append({"type": "text", "text": f"Image F{number}:"})
            content.append(_image_block(figure["image"]))
        pages = sorted({f["page"] for f in batch})
        reply = _call_batch(call, content, f"describe the figures on page(s) {pages}", warnings)
        if reply is None:
            continue
        replies = {r.get("id"): r for r in reply.get("figures", []) if isinstance(r, dict)}
        for number, figure in enumerate(batch, start=1):
            reply = replies.get(f"F{number}")
            if reply and str(reply.get("description", "")).strip():
                figure["description"] = str(reply["description"]).strip()
                figure["data"] = str(reply.get("data") or "").strip()


def summarize_tables(tables, call, filename, warnings):
    for batch in _table_batches(tables):
        items = []
        for number, table in enumerate(batch, start=1):
            caption = f" (caption: {table['caption']})" if table.get("caption") else ""
            items.append(f"T{number}{caption}:\n{table['text'][:TABLE_MAX_CHARS_IN_PROMPT]}")
        pages = sorted({t["page"] for t in batch})
        reply = _call_batch(call, [{"type": "text", "text": TABLE_PROMPT.format(
            filename=filename, items="\n\n".join(items))}],
            f"summarize the tables on page(s) {pages}", warnings)
        if reply is None:
            continue
        replies = {r.get("id"): r for r in reply.get("tables", []) if isinstance(r, dict)}
        for number, table in enumerate(batch, start=1):
            summary = str((replies.get(f"T{number}") or {}).get("summary", "")).strip()
            if summary:
                table["summary"] = summary


def summarize_sections(layout, call, filename, warnings):
    found = sorted(chapters(build_sections(layout["elements"])),
                   key=lambda c: len(c["text"]), reverse=True)[:config.ENRICHMENT_MAX_SECTIONS]
    found.sort(key=lambda c: c["page_start"])  # back to reading order
    summaries = layout.setdefault("section_summaries", [])
    done = {s["path_text"] for s in summaries}
    for chapter in found:
        if chapter["path_text"] in done:
            continue
        reply = _call_batch(call, [{"type": "text", "text": CHAPTER_PROMPT.format(
            filename=filename, title=chapter["path_text"],
            text=chapter["text"][:CHAPTER_MAX_CHARS_IN_PROMPT])}],
            f"summarize '{chapter['path_text']}'", warnings)
        if reply and str(reply.get("summary", "")).strip():
            summaries.append({key: chapter[key] for key in ("path_text", "page_start", "page_end")}
                             | {"summary": str(reply["summary"]).strip()})

    if summaries and not layout.get("document_summary"):
        text = "\n\n".join(f"{s['path_text']}: {s['summary']}" for s in summaries)
        reply = _call_batch(call, [{"type": "text", "text": DOCUMENT_PROMPT.format(
            filename=filename, text=text)}], "summarize the whole document", warnings)
        if reply and str(reply.get("summary", "")).strip():
            layout["document_summary"] = str(reply["summary"]).strip()


def enrich_layout(layout, profile, filename, llm=None, sleep=time.sleep, on_status=None):
    """Add Gemini descriptions and summaries to a parsed document. Returns warnings."""
    if not config.ENRICHMENT_ENABLED:
        return []
    elements = layout["elements"]
    figures = [e for e in elements if e["type"] == "picture" and e.get("image")
               and "description" not in e and Path(e["image"]).exists()] \
        if profile_setting(profile, "describe_figures") else []
    tables = [e for e in elements if e["type"] == "table" and "summary" not in e] \
        if profile_setting(profile, "summarize_tables") else []
    sections = profile_setting(profile, "summarize_sections") and not layout.get("document_summary")

    # Items already done count toward the limits too
    done_figures = sum(1 for e in elements if e["type"] == "picture" and "description" in e)
    done_tables = sum(1 for e in elements if e["type"] == "table" and "summary" in e)
    figures = figures[:max(0, config.ENRICHMENT_MAX_FIGURES - done_figures)]
    tables = tables[:max(0, config.ENRICHMENT_MAX_TABLES - done_tables)]
    if not (figures or tables or sections):
        return []
    if not is_embedding_configured():  # the same GOOGLE_API_KEY
        return ["Gemini enrichment was skipped because GOOGLE_API_KEY is not set."]
    if on_status:
        on_status("enriching")

    call = GeminiCaller(llm or create_enrichment_model(), sleep=sleep)
    warnings = []
    try:
        if figures:
            describe_figures(figures, call, filename, warnings)
        if tables:
            summarize_tables(tables, call, filename, warnings)
        if sections:
            summarize_sections(layout, call, filename, warnings)
    except StopEnrichment as err:
        warnings.append(f"Enrichment stopped early: {err} Process the document again later "
                        "with force=true to finish it.")
    missing = sum(1 for e in figures if "description" not in e) + \
        sum(1 for e in tables if "summary" not in e)
    if missing:
        warnings.append(f"{missing} figure(s)/table(s) could not be enriched; they are still "
                        "searchable by their text and captions.")
    return warnings
