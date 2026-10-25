"""Chunks a Docling-parsed document by its structure (structured and visual profiles).

app.chunking splits plain page text. This module instead uses the elements
from app.parsing, so that:
  - chunks follow sections (a chunk never mixes two sections, but may span
    pages) and know their section path, e.g. "2 Outlook > 2.1 Inflation";
  - every table is its own chunk, kept whole or split by rows with the
    header repeated, with its caption and Gemini summary;
  - every chart or diagram Gemini described becomes a figure chunk;
  - visual documents get footnote chunks, and footnotes are attached to the
    tables and figures on their page;
  - structured documents get parent/child chunks: small child chunks are
    embedded and searched, and their whole parent section is what the chat
    model reads. Chapter and document summaries become summary chunks.

Chunks are returned as dicts in the same shape as chunking.chunk_to_dict().
"""

from collections import defaultdict

from app.chunking import create_splitter
from app.parsing import render_element
from app.profiles import profile_setting

PARENT_SIZE = 3000
CHILD_SIZE = 600
CHILD_OVERLAP = 100
TABLE_MAX_CHARS = 4000
PAGE_FOOTNOTES_MAX_CHARS = 1500
HEADING_MAX_CHARS = 80
# Chapters with less text than this are not summarized
CHAPTER_MIN_CHARS = 3000

FLOW_TYPES = ("text", "list_item", "code", "formula")


def _short(text):
    text = " ".join(text.split())
    return text if len(text) <= HEADING_MAX_CHARS else text[:HEADING_MAX_CHARS - 3].rstrip() + "..."


def build_sections(elements):
    """Group elements under their headings.

    Returns a list of sections in reading order, each a dict with "path"
    (headings from the top level down), "path_text", "elements",
    "page_start" and "page_end". Sections without content are left out.
    """
    sections = [{"path": [], "elements": []}]
    stack = []  # (level, heading text) of the current heading and its parents
    for element in elements:
        if element["type"] == "heading":
            level = element["level"] or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _short(element["text"])))
            sections.append({"path": [text for _, text in stack], "elements": []})
        else:
            sections[-1]["elements"].append(element)

    result = [s for s in sections if s["elements"]]
    for section in result:
        pages = [e["page"] for e in section["elements"]]
        section.update(path_text=" > ".join(section["path"]) or None,
                       page_start=min(pages), page_end=max(pages))
    return result


def chapters(sections):
    """Group sections by their top-level heading, for chapter summaries.

    Returns [{"path_text", "text", "page_start", "page_end"}] for chapters
    with at least CHAPTER_MIN_CHARS of text.
    """
    groups = []
    for section in sections:
        top = section["path"][0] if section["path"] else None
        if not groups or groups[-1]["title"] != top:
            groups.append({"title": top, "sections": []})
        groups[-1]["sections"].append(section)

    result = []
    for group in groups:
        texts = [render_element(e) for s in group["sections"] for e in s["elements"]
                 if e["type"] in FLOW_TYPES or e["type"] == "table"]
        text = "\n\n".join(t for t in texts if t.strip())
        if len(text) >= CHAPTER_MIN_CHARS:
            result.append({
                "path_text": group["title"] or "Introduction",
                "text": text,
                "page_start": min(s["page_start"] for s in group["sections"]),
                "page_end": max(s["page_end"] for s in group["sections"]),
            })
    return result


def split_table(markdown, max_chars=TABLE_MAX_CHARS):
    """Split a long markdown table by rows, repeating the header in every part."""
    lines = markdown.splitlines()
    if len(markdown) <= max_chars or len(lines) < 4:
        return [markdown]
    header, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > max_chars:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    parts.append("\n".join(header + current))
    return parts


class _ChunkList:
    """Collects chunks with consistent ids, positions and metadata."""

    def __init__(self, document_id, filename, printed_pages):
        self.document_id = document_id
        self.filename = filename
        self.printed_pages = printed_pages
        self.chunks = []
        self.counters = defaultdict(int)

    def add(self, kind, text, page, page_end, chunk_type, section_path,
            parent_chunk_id=None, embed=True):
        text = text.strip()
        if not text:
            return None
        if parent_chunk_id:
            self.counters[parent_chunk_id] += 1
            chunk_id = f"{parent_chunk_id}_c{self.counters[parent_chunk_id]}"
        else:
            self.counters[kind] += 1
            chunk_id = f"{self.document_id}_{kind}{self.counters[kind]}"
        metadata = {
            "source": self.filename,
            "page": page,
            "page_end": page_end,
            "chunk_id": chunk_id,
            "chunk_index": len(self.chunks),
            "document_id": self.document_id,
            "chunk_type": chunk_type,
            "section_path": section_path,
            "parent_chunk_id": parent_chunk_id,
            "embed": embed,
        }
        printed = self.printed_pages.get(page)
        if printed and printed != str(page):
            metadata["printed_page"] = printed
        self.chunks.append({"chunk_id": chunk_id, "text": text, "char_count": len(text),
                            "metadata": metadata})
        return chunk_id


def _flow_blocks(elements, max_size):
    """Group consecutive flow elements into blocks of about max_size characters.

    Yields (text, first_page, last_page). An element longer than max_size
    is a block of its own (split later).
    """
    block = []
    size = 0
    for element in elements:
        text = render_element(element)
        if block and size + len(text) > max_size:
            yield "\n\n".join(t for _, t in block), block[0][0]["page"], block[-1][0]["page"]
            block, size = [], 0
        block.append((element, text))
        size += len(text) + 2
    if block:
        yield "\n\n".join(t for _, t in block), block[0][0]["page"], block[-1][0]["page"]


def _with_footnotes(text, footnotes):
    if not footnotes:
        return text
    notes = "\n".join(f"- {note}" for note in footnotes)
    if len(notes) > PAGE_FOOTNOTES_MAX_CHARS:
        notes = notes[:PAGE_FOOTNOTES_MAX_CHARS].rstrip() + " ..."
    return f"{text}\n\nFootnotes on this page:\n{notes}"


def chunk_layout(layout, document_id, filename, profile, chunk_size, chunk_overlap):
    """Chunk a parsed document (app.parsing.parse_pdf result) for a profile."""
    parent_child = profile_setting(profile, "parent_child")
    footnote_chunks = profile_setting(profile, "footnote_chunks")
    printed = {p["page_number"]: p.get("printed_page") for p in layout["pages"]}
    out = _ChunkList(document_id, filename, printed)
    splitter = create_splitter(chunk_size, chunk_overlap)
    child_splitter = create_splitter(CHILD_SIZE, CHILD_OVERLAP)
    parent_splitter = create_splitter(PARENT_SIZE, 0)

    footnotes_by_page = defaultdict(list)
    if footnote_chunks:
        for element in layout["elements"]:
            if element["type"] == "footnote":
                footnotes_by_page[element["page"]].append(element["text"])

    def add_flow(buffer, section_path):
        for text, first, last in _flow_blocks(buffer, PARENT_SIZE if parent_child else chunk_size):
            if parent_child:
                # A single very long element can make a block larger than a parent may be
                parents = [text] if len(text) <= PARENT_SIZE else parent_splitter.split_text(text)
                for parent_text in parents:
                    parent_id = out.add("p", parent_text, first, last, "text", section_path,
                                        embed=False)
                    for piece in child_splitter.split_text(parent_text):
                        out.add("c", piece, first, last, "text", section_path,
                                parent_chunk_id=parent_id)
            else:
                pieces = [text] if len(text) <= chunk_size else splitter.split_text(text)
                for piece in pieces:
                    out.add("t", piece, first, last, "text", section_path)

    for section in build_sections(layout["elements"]):
        path = section["path_text"]
        buffer = []
        for element in section["elements"]:
            if element["type"] in FLOW_TYPES or (element["type"] == "footnote" and not footnote_chunks):
                buffer.append(element)
                continue
            add_flow(buffer, path)
            buffer = []
            page, notes = element["page"], footnotes_by_page.get(element["page"])

            if element["type"] == "table":
                parts = split_table(element["text"])
                for number, part in enumerate(parts, start=1):
                    lines = [f"Table: {element['caption']}" if element.get("caption") else "",
                             f"Summary: {element['summary']}" if element.get("summary") else "",
                             f"(Part {number} of {len(parts)})" if len(parts) > 1 else "",
                             part]
                    text = "\n".join(line for line in lines if line)
                    out.add("tab", _with_footnotes(text, notes), page, page, "table", path)
            elif element["type"] == "picture":
                if not (element.get("description") or element.get("caption")):
                    continue  # nothing known about it
                lines = [f"Figure: {element['caption']}" if element.get("caption") else "Figure",
                         element.get("description") or "",
                         f"Data read from the chart:\n{element['data']}" if element.get("data") else ""]
                text = "\n".join(line for line in lines if line)
                out.add("fig", _with_footnotes(text, notes), page, page, "figure", path)
            elif element["type"] == "footnote":
                out.add("fn", f"Footnote: {element['text']}", page, page, "footnote", path)
        add_flow(buffer, path)

    # A cached layout may hold summaries made for the structured profile
    if not profile_setting(profile, "summarize_sections"):
        return out.chunks
    for summary in layout.get("section_summaries") or []:
        out.add("sum", f"Summary of {summary['path_text']}:\n{summary['summary']}",
                summary["page_start"], summary["page_end"], "summary", summary["path_text"])
    if layout.get("document_summary"):
        out.add("sum", f"Summary of the whole document:\n{layout['document_summary']}",
                1, len(layout["pages"]), "summary", None)
    return out.chunks
