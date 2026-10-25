"""Tests for cleaning Docling elements and rebuilding page text.

All tests but the last run without Docling. The last one runs real Docling on
a generated PDF; it is slow (and downloads models the first time), so it only
runs with RUN_INTEGRATION_TESTS=1.
"""

import os

import pytest

from app.parsing import (
    build_pages,
    clean_elements,
    drop_repeated_lines,
    find_listing_pages,
    parse_pdf,
    printed_page_labels,
    render_element,
)
from pdf_helpers import make_pdf


def el(type_, text, page, level=None, caption=None):
    return {"type": type_, "text": text, "page": page, "level": level, "bbox": None,
            "caption": caption}


# ---------- Running headers and footers ----------

def test_lines_repeated_on_most_pages_are_removed():
    elements = []
    for page in range(1, 11):
        elements += [el("text", "Global Economic Prospects | June 2025", page),
                     el("text", f"Page {page}", page),
                     el("text", f"Unique paragraph about topic {page} with details.", page)]

    cleaned = drop_repeated_lines(elements, page_count=10)

    assert [e["text"] for e in cleaned] == [
        f"Unique paragraph about topic {n} with details." for n in range(1, 11)]


def test_long_paragraphs_and_tables_are_never_treated_as_repeated():
    long_text = "This sentence is long enough that it cannot be a running header. " * 3
    elements = [el("text", long_text, p) for p in range(1, 6)]
    elements += [el("table", "| a | b |", p) for p in range(1, 6)]

    assert drop_repeated_lines(elements, page_count=5) == elements


def test_short_documents_keep_everything():
    elements = [el("text", "Header", 1), el("text", "Header", 2)]

    assert drop_repeated_lines(elements, page_count=2) == elements


# ---------- Contents and index pages ----------

def contents_lines(page):
    return [el("text", f"Chapter {n} Topic name ........ {n * 10}", page) for n in range(1, 7)]


def test_contents_page_with_a_heading_is_found():
    elements = [el("heading", "Contents", 2, level=1), el("text", "Foreword", 2)]
    elements += [el("text", "Real text.", p) for p in range(3, 21)]

    assert find_listing_pages(elements, page_count=20) == {2}


def test_contents_page_without_a_heading_is_found_by_its_lines():
    elements = contents_lines(3) + [el("text", "Real text.", p) for p in range(4, 21)]

    assert find_listing_pages(elements, page_count=20) == {3}


def test_index_at_the_back_is_found():
    index = [el("text", f"Inflation, {n}, {n + 40}", 20) for n in range(1, 8)]
    elements = [el("text", "Real text.", p) for p in range(1, 20)] + index

    assert find_listing_pages(elements, page_count=20) == {20}


def test_listing_in_the_middle_of_the_document_is_kept():
    # A numbered list in the middle is content, not a table of contents
    elements = [el("text", "Real text.", p) for p in range(1, 21) if p != 10] + contents_lines(10)

    assert find_listing_pages(elements, page_count=20) == set()


def test_clean_elements_reports_what_was_removed():
    elements = [el("heading", "Contents", 1, level=1)]
    for page in range(1, 11):
        elements += [el("text", "Annual Report 2024", page),
                     el("text", f"Body text for page {page}.", page)]

    cleaned, removed = clean_elements(elements, page_count=10)

    assert removed == {"repeated_lines": 10, "listing_pages": [1]}
    assert [e["text"] for e in cleaned] == [f"Body text for page {n}." for n in range(2, 11)]


# ---------- Rebuilding pages ----------

def test_elements_render_as_markdown():
    assert render_element(el("heading", "Inflation", 1, level=2)) == "## Inflation"
    assert render_element(el("list_item", "First point", 1)) == "- First point"
    assert render_element(el("list_item", "1. First point", 1)) == "1. First point"
    assert render_element(el("table", "| a | b |", 1, caption="Table 2.1 Growth")) == (
        "Table: Table 2.1 Growth\n| a | b |")
    assert render_element(el("picture", "", 1, caption="Figure 3 Emissions")) == (
        "Figure: Figure 3 Emissions")
    assert render_element(el("picture", "", 1)) == ""


def test_build_pages_keeps_every_page_and_printed_labels():
    elements = [el("heading", "Overview", 1, level=1), el("text", "First page text.", 1),
                el("text", "Third page text.", 3)]

    pages = build_pages(elements, "wb.pdf", page_count=3, printed_pages=["i", "ii", "1"])

    assert [p["page_number"] for p in pages] == [1, 2, 3]
    assert pages[0]["text"] == "# Overview\n\nFirst page text."
    assert pages[1]["text"] == "" and pages[1]["char_count"] == 0
    assert [p["printed_page"] for p in pages] == ["i", "ii", "1"]
    assert all(p["filename"] == "wb.pdf" for p in pages)


def test_empty_pages_need_ocr_unless_removed_on_purpose():
    elements = [el("text", "A page with plenty of real text on it.", 1)]

    pages = build_pages(elements, "wb.pdf", page_count=3, removed_pages={2})

    assert [p["needs_ocr"] for p in pages] == [False, False, True]


class FakeReader:
    def __init__(self, page_count, labels):
        self.pages = [object()] * page_count
        self.page_labels = labels


def test_printed_page_labels():
    assert printed_page_labels(FakeReader(3, ["i", "ii", "1"])) == ["i", "ii", "1"]
    assert printed_page_labels(FakeReader(2, [])) == ["1", "2"]  # no labels defined


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_INTEGRATION_TESTS") != "1", reason="Set RUN_INTEGRATION_TESTS=1")
def test_real_docling_parses_a_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(make_pdf(["The quick brown fox jumps over the lazy dog.",
                               "A second page with different words entirely."]))

    result = parse_pdf(path, "doc.pdf", page_count=2)

    assert [p["page_number"] for p in result["pages"]] == [1, 2]
    assert "quick brown fox" in result["pages"][0]["text"]
    assert "second page" in result["pages"][1]["text"]
    assert all(e["page"] in (1, 2) and e["bbox"] for e in result["elements"])
