"""Tests for structure-aware chunking of Docling elements (structured and visual profiles)."""

from app.chunkers import CHAPTER_MIN_CHARS, build_sections, chapters, chunk_layout, split_table


def el(type_, text, page, level=None, caption=None, **extra):
    return {"type": type_, "text": text, "page": page, "level": level, "bbox": None,
            "caption": caption, **extra}


def layout(elements, page_count=5, **extra):
    pages = [{"page_number": n, "printed_page": str(n)} for n in range(1, page_count + 1)]
    return {"elements": elements, "pages": pages, **extra}


def chunks_of(chunks, chunk_type):
    return [c for c in chunks if c["metadata"]["chunk_type"] == chunk_type]


SAMPLE = [
    el("heading", "1 Climate", 1, level=1),
    el("text", "We cut emissions across the product line.", 1),
    el("heading", "1.1 Carbon footprint", 2, level=2),
    el("text", "The footprint of each iMac fell.", 2),
    el("table", "| Year | kg CO2e |\n|---|---|\n| 2015 | 900 |\n| 2024 | 600 |", 2,
       caption="Table 1 Footprint", summary="Footprint fell from 900 to 600 kg."),
    el("picture", "", 3, caption="Figure 2 Energy use", description="A bar chart of energy use.",
       data="| Year | kWh |\n|---|---|\n| 2024 | 50 |"),
    el("picture", "", 3),  # nothing known about it: no chunk
    el("footnote", "1. Compared with the 2015 baseline.", 2),
    el("heading", "2 Resources", 4, level=1),
    el("list_item", "Recycled aluminium in the stand", 4),
]


# ---------- Sections ----------

def test_sections_follow_heading_levels():
    sections = build_sections(SAMPLE)

    assert [s["path_text"] for s in sections] == [
        "1 Climate", "1 Climate > 1.1 Carbon footprint", "2 Resources"]
    assert (sections[1]["page_start"], sections[1]["page_end"]) == (2, 3)


def test_text_before_the_first_heading_has_no_section():
    sections = build_sections([el("text", "Cover text", 1), el("heading", "Intro", 2, level=1),
                               el("text", "Body", 2)])

    assert sections[0]["path_text"] is None
    assert sections[1]["path"] == ["Intro"]


def test_long_headings_are_shortened_in_paths():
    sections = build_sections([el("heading", "word " * 40, 1, level=1), el("text", "Body", 1)])

    assert len(sections[0]["path_text"]) <= 80
    assert sections[0]["path_text"].endswith("...")


def test_chapters_group_sections_by_top_level_heading():
    long = "A sentence about the economy. " * (CHAPTER_MIN_CHARS // 25)
    elements = [el("heading", "1 Outlook", 1, level=1), el("text", long, 1),
                el("heading", "1.1 Prices", 2, level=2), el("text", long, 3),
                el("heading", "2 Short chapter", 4, level=1), el("text", "Too short.", 4)]

    found = chapters(build_sections(elements))

    assert [c["path_text"] for c in found] == ["1 Outlook"]
    assert (found[0]["page_start"], found[0]["page_end"]) == (1, 3)


# ---------- Visual profile ----------

def test_visual_chunks_tables_figures_and_footnotes_separately():
    chunks = chunk_layout(layout(SAMPLE), "doc", "imac.pdf", "visual", 1000, 150)

    [table] = chunks_of(chunks, "table")
    assert table["text"].startswith("Table: Table 1 Footprint\nSummary: Footprint fell")
    assert "| 2024 | 600 |" in table["text"]
    # Footnotes on the same page are attached to the table
    assert "Footnotes on this page:\n- 1. Compared with the 2015 baseline." in table["text"]
    assert table["metadata"]["section_path"] == "1 Climate > 1.1 Carbon footprint"

    [figure] = chunks_of(chunks, "figure")
    assert figure["text"].startswith("Figure: Figure 2 Energy use\nA bar chart of energy use.")
    assert "Data read from the chart:\n| Year | kWh |" in figure["text"]

    [footnote] = chunks_of(chunks, "footnote")
    assert footnote["text"] == "Footnote: 1. Compared with the 2015 baseline."

    texts = chunks_of(chunks, "text")
    assert [c["metadata"]["section_path"] for c in texts] == [
        "1 Climate", "1 Climate > 1.1 Carbon footprint", "2 Resources"]
    assert texts[2]["text"] == "- Recycled aluminium in the stand"
    assert all(c["metadata"]["embed"] for c in chunks)


def test_chunk_ids_positions_and_metadata():
    chunks = chunk_layout(layout(SAMPLE), "doc", "imac.pdf", "visual", 1000, 150)

    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        meta = chunk["metadata"]
        assert chunk["chunk_id"] == meta["chunk_id"] and chunk["chunk_id"].startswith("doc_")
        assert meta["source"] == "imac.pdf" and meta["document_id"] == "doc"
        assert meta["page"] <= meta["page_end"]
        assert chunk["char_count"] == len(chunk["text"])


def test_printed_page_labels_are_kept_when_they_differ():
    pages = [{"page_number": 1, "printed_page": "iv"}]
    chunks = chunk_layout({"elements": [el("text", "Foreword text.", 1)], "pages": pages},
                          "doc", "wb.pdf", "visual", 1000, 150)

    assert chunks[0]["metadata"]["printed_page"] == "iv"


def test_long_text_in_one_section_is_split_but_keeps_its_section():
    elements = [el("heading", "Overview", 1, level=1)] + [
        el("text", f"Paragraph {n} " + "about growth. " * 20, 1 + n // 3) for n in range(10)]

    texts = chunks_of(chunk_layout(layout(elements), "doc", "x.pdf", "visual", 1000, 150), "text")

    assert len(texts) > 1
    assert all(len(c["text"]) <= 1000 for c in texts)
    assert {c["metadata"]["section_path"] for c in texts} == {"Overview"}
    assert texts[-1]["metadata"]["page_end"] == 4  # chunks may span pages


def test_large_tables_are_split_with_the_header_repeated():
    rows = "\n".join(f"| Row {n} | {n * 10} |" for n in range(400))
    markdown = f"| Item | Value |\n|---|---|\n{rows}"

    parts = split_table(markdown, max_chars=1000)

    assert len(parts) > 1
    assert all(p.startswith("| Item | Value |\n|---|---|\n") and len(p) <= 1000 for p in parts)
    assert sum(p.count("| Row ") for p in parts) == 400


def test_split_table_parts_are_labelled():
    rows = "\n".join(f"| Row {n} | {n} |" for n in range(600))
    elements = [el("table", f"| A | B |\n|---|---|\n{rows}", 1, caption="Big table")]

    tables = chunks_of(chunk_layout(layout(elements), "doc", "x.pdf", "visual", 1000, 150), "table")

    assert len(tables) > 1
    assert "(Part 1 of" in tables[0]["text"] and tables[0]["text"].startswith("Table: Big table")


# ---------- Structured profile ----------

def test_structured_uses_parent_and_child_chunks():
    elements = [el("heading", "2 Outlook", 1, level=1)] + [
        el("text", f"Paragraph {n}. " + "Growth slowed in many economies. " * 12, 1) for n in range(6)]

    chunks = chunk_layout(layout(elements), "doc", "wb.pdf", "structured", 1000, 150)

    parents = [c for c in chunks if not c["metadata"]["embed"]]
    children = [c for c in chunks if c["metadata"]["embed"]]
    assert parents and children
    assert all(len(p["text"]) <= 3000 for p in parents)
    assert all(len(c["text"]) <= 600 for c in children)
    parent_ids = {p["chunk_id"] for p in parents}
    assert all(c["metadata"]["parent_chunk_id"] in parent_ids for c in children)
    assert all(c["chunk_id"].startswith(c["metadata"]["parent_chunk_id"] + "_c") for c in children)
    # Every parent's text is covered by its children
    for parent in parents:
        own = [c for c in children if c["metadata"]["parent_chunk_id"] == parent["chunk_id"]]
        assert own[0]["text"][:40] == parent["text"][:40]


def test_structured_keeps_footnotes_in_the_text_flow():
    elements = [el("heading", "Box 1", 1, level=1), el("text", "Prices rose.", 1),
                el("footnote", "Source: World Bank.", 1)]

    chunks = chunk_layout(layout(elements), "doc", "wb.pdf", "structured", 1000, 150)

    assert chunks_of(chunks, "footnote") == []
    assert "Source: World Bank." in chunks[0]["text"]


def test_section_and_document_summaries_become_summary_chunks():
    summaries = [{"path_text": "2 Outlook", "page_start": 3, "page_end": 9,
                  "summary": "Growth is slowing."}]
    data = layout([el("text", "Body.", 1)], page_count=10, section_summaries=summaries,
                  document_summary="A report on the world economy.")

    found = chunks_of(chunk_layout(data, "doc", "wb.pdf", "structured", 1000, 150), "summary")

    assert [c["text"] for c in found] == ["Summary of 2 Outlook:\nGrowth is slowing.",
                                          "Summary of the whole document:\nA report on the world economy."]
    assert (found[0]["metadata"]["page"], found[0]["metadata"]["page_end"]) == (3, 9)
    assert (found[1]["metadata"]["page"], found[1]["metadata"]["page_end"]) == (1, 10)


def test_visual_chunks_ignore_summaries_cached_for_the_structured_profile():
    data = layout([el("text", "Body.", 1)], section_summaries=[
        {"path_text": "2 Outlook", "page_start": 1, "page_end": 1, "summary": "S."}],
        document_summary="D.")

    chunks = chunk_layout(data, "doc", "x.pdf", "visual", 1000, 150)

    assert chunks_of(chunks, "summary") == []


def test_one_huge_element_becomes_several_parents():
    elements = [el("heading", "Annex", 1, level=1),
                el("text", "A long sentence about statistics and data. " * 220, 1)]  # ~9,000 chars

    chunks = chunk_layout(layout(elements), "doc", "wb.pdf", "structured", 1000, 150)

    parents = [c for c in chunks if not c["metadata"]["embed"]]
    assert len(parents) >= 3
    assert all(len(p["text"]) <= 3000 for p in parents)
