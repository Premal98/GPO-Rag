"""Tests for Gemini enrichment, with the Gemini model replaced by a fake."""

import json
import re

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from app import config
from app.chunkers import CHAPTER_MIN_CHARS
from app.enrichment import GeminiCaller, enrich_layout, parse_json, EnrichmentError


class FakeGemini:
    """Answers each enrichment prompt in the expected JSON shape, or raises given errors."""

    def __init__(self, errors=None):
        self.errors = list(errors or [])
        self.calls = []  # the content list of each request

    def invoke(self, messages):
        content = messages[0].content
        self.calls.append(content)
        if self.errors:
            raise self.errors.pop(0)
        prompt = content[0]["text"]
        if '"figures"' in prompt:
            ids = re.findall(r"^F(\d+):", prompt, re.MULTILINE)
            reply = {"figures": [{"id": f"F{i}", "description": f"Chart {i}.",
                                  "data": "| a | 1 |"} for i in ids]}
        elif '"tables"' in prompt:
            ids = re.findall(r"^T(\d+)", prompt, re.MULTILINE)
            reply = {"tables": [{"id": f"T{i}", "summary": f"Table summary {i}."} for i in ids]}
        elif "Part:" in prompt:
            title = re.search(r"^Part: (.*)$", prompt, re.MULTILINE).group(1)
            reply = {"summary": f"About {title}."}
        else:
            reply = {"summary": "The whole document."}
        return AIMessage(content=json.dumps(reply))


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(config, "ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(config, "ENRICHMENT_MIN_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(config, "ENRICHMENT_MAX_FIGURES", 40)
    monkeypatch.setattr(config, "ENRICHMENT_MAX_TABLES", 80)
    monkeypatch.setattr(config, "ENRICHMENT_MAX_SECTIONS", 30)


def no_sleep(seconds):
    pass


def image(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (200, 150), "white").save(path)
    return path.as_posix()


def el(type_, text, page, **extra):
    return {"type": type_, "text": text, "page": page, "level": extra.pop("level", None),
            "bbox": None, "caption": extra.pop("caption", None), **extra}


def make_layout(tmp_path, figures=2, tables=3):
    elements = [el("picture", "", 1, caption=f"Figure {n}", image=image(tmp_path, f"f{n}.png"))
                for n in range(figures)]
    elements += [el("table", f"| a | b |\n|---|---|\n| {n} | 2 |", 2) for n in range(tables)]
    return {"elements": elements, "pages": [{"page_number": 1}, {"page_number": 2}]}


def enrich(layout, profile="visual", llm=None):
    return enrich_layout(layout, profile, "report.pdf", llm=llm or FakeGemini(), sleep=no_sleep)


# ---------- What gets enriched ----------

def test_figures_get_descriptions_and_tables_get_summaries(tmp_path):
    layout = make_layout(tmp_path, figures=5, tables=3)
    llm = FakeGemini()

    warnings = enrich(layout, llm=llm)

    assert warnings == []
    figures = [e for e in layout["elements"] if e["type"] == "picture"]
    assert [f["description"] for f in figures] == ["Chart 1.", "Chart 2.", "Chart 3.", "Chart 4.",
                                                   "Chart 1."]
    assert all(f["data"] == "| a | 1 |" for f in figures)
    tables = [e for e in layout["elements"] if e["type"] == "table"]
    assert [t["summary"] for t in tables] == ["Table summary 1.", "Table summary 2.",
                                              "Table summary 3."]
    # 5 figures in batches of 4, and the 3 small tables in one request
    assert len(llm.calls) == 3


def test_images_are_sent_with_the_figure_prompt(tmp_path):
    llm = FakeGemini()

    enrich(make_layout(tmp_path, figures=2, tables=0), llm=llm)

    content = llm.calls[0]
    images = [block for block in content if block["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "F1: page 1, caption: Figure 0" in content[0]["text"]


def test_structured_documents_also_get_chapter_and_document_summaries(tmp_path):
    long = "The economy grew in many regions this year. " * (CHAPTER_MIN_CHARS // 40)
    layout = {"elements": [el("heading", "1 Outlook", 1, level=1), el("text", long, 1),
                           el("heading", "2 Risks", 2, level=1), el("text", long, 2)],
              "pages": [{"page_number": 1}, {"page_number": 2}]}

    enrich(layout, profile="structured")

    assert [(s["path_text"], s["summary"]) for s in layout["section_summaries"]] == [
        ("1 Outlook", "About 1 Outlook."), ("2 Risks", "About 2 Risks.")]
    assert layout["document_summary"] == "The whole document."


def test_visual_documents_get_no_section_summaries(tmp_path):
    long = "Text. " * CHAPTER_MIN_CHARS
    layout = {"elements": [el("heading", "1 Climate", 1, level=1), el("text", long, 1)],
              "pages": [{"page_number": 1}]}

    enrich(layout, profile="visual")

    assert "section_summaries" not in layout


def test_narrative_documents_are_not_enriched(tmp_path):
    llm = FakeGemini()

    assert enrich(make_layout(tmp_path), profile="narrative", llm=llm) == []
    assert llm.calls == []


def test_items_done_before_are_not_sent_again(tmp_path):
    layout = make_layout(tmp_path)
    enrich(layout)
    llm = FakeGemini()

    enrich(layout, llm=llm)

    assert llm.calls == []


def test_limits_cap_how_many_items_are_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENRICHMENT_MAX_FIGURES", 1)
    monkeypatch.setattr(config, "ENRICHMENT_MAX_TABLES", 0)
    layout = make_layout(tmp_path, figures=3, tables=2)

    enrich(layout)

    assert sum("description" in e for e in layout["elements"]) == 1
    assert not any("summary" in e for e in layout["elements"])


def test_figures_without_a_saved_image_are_skipped(tmp_path):
    layout = make_layout(tmp_path, figures=0, tables=0)
    layout["elements"].append(el("picture", "", 1, caption="Logo", image=None))
    llm = FakeGemini()

    enrich(layout, llm=llm)

    assert llm.calls == []


# ---------- Settings and failures ----------

def test_disabled_enrichment_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENRICHMENT_ENABLED", False)
    llm = FakeGemini()

    assert enrich(make_layout(tmp_path), llm=llm) == []
    assert llm.calls == []


def test_missing_api_key_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

    warnings = enrich(make_layout(tmp_path))

    assert "GOOGLE_API_KEY" in warnings[0]


def test_a_failed_batch_is_a_warning_and_the_rest_continues(tmp_path):
    llm = FakeGemini(errors=[RuntimeError("500 INTERNAL")])
    layout = make_layout(tmp_path, figures=2, tables=2)

    warnings = enrich(layout, llm=llm)

    assert "Could not describe the figures on page(s) [1]" in warnings[0]
    assert all("summary" in e for e in layout["elements"] if e["type"] == "table")
    assert "2 figure(s)/table(s) could not be enriched" in warnings[-1]


def test_rate_limits_are_retried_after_the_suggested_wait(tmp_path):
    waits = []
    llm = FakeGemini(errors=[RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 7s.")])
    layout = make_layout(tmp_path, figures=0, tables=1)

    warnings = enrich_layout(layout, "visual", "r.pdf", llm=llm, sleep=waits.append)

    assert warnings == []
    assert waits == [8.0]
    assert layout["elements"][0]["summary"] == "Table summary 1."


def test_daily_quota_stops_enrichment_with_a_hint(tmp_path):
    llm = FakeGemini(errors=[RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay")])
    layout = make_layout(tmp_path, figures=2, tables=2)

    warnings = enrich(layout, llm=llm)

    assert len(llm.calls) == 1
    assert "stopped early" in warnings[0] and "force=true" in warnings[0]


def test_status_is_reported_only_when_requests_are_made(tmp_path):
    stages = []
    enrich_layout(make_layout(tmp_path), "visual", "r.pdf", llm=FakeGemini(), sleep=no_sleep,
                  on_status=stages.append)
    enrich_layout(make_layout(tmp_path, figures=0, tables=0), "visual", "r.pdf",
                  llm=FakeGemini(), sleep=no_sleep, on_status=stages.append)

    assert stages == ["enriching"]


def test_requests_are_spaced_out(monkeypatch):
    monkeypatch.setattr(config, "ENRICHMENT_MIN_INTERVAL_SECONDS", 4)
    waits, now = [], [100.0]
    caller = GeminiCaller(FakeGemini(), sleep=waits.append, clock=lambda: now[0])
    prompt = [{"type": "text", "text": "Reply with JSON only: {\"summary\": \"...\"}"}]

    caller(prompt)
    now[0] += 1
    caller(prompt)

    assert waits == [3.0]


def test_replies_must_be_json_objects():
    assert parse_json('```json\n{"summary": "x"}\n```') == {"summary": "x"}
    with pytest.raises(EnrichmentError):
        parse_json("not json")
    with pytest.raises(EnrichmentError):
        parse_json("[1, 2]")
