"""Layout-aware PDF parsing with Docling, for structured and visual documents.

pypdf (app.pdf_ingestion) gives flat text per page. Docling instead finds
the layout: headings with levels, paragraphs, tables, pictures, captions and
footnotes, each with its page and position. This module turns Docling's
output into a plain list of "elements", removes page furniture (running
headers and footers, tables of contents, indexes), and rebuilds page text
from what is left so the existing chunker can use it.

Each element is a dict:
  type     heading | text | list_item | table | picture | footnote | code | formula
  text     the text; for tables, a markdown table
  page     1-based page number
  level    heading level (1 = top), else None
  bbox     [left, top, right, bottom] in PDF points from the page's top-left, or None
  caption  caption of a table or picture, else None

Docling is imported only when a PDF is actually parsed, because loading it
(and torch) takes several seconds.
"""

import os
import re
import threading
import time
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from app.pdf_ingestion import MIN_CHARS_PER_PAGE

# Docling label -> element type. Labels not listed here are page furniture
# (page_header, page_footer, document_index, ...) and are dropped.
LABEL_TYPES = {
    "title": "heading",
    "section_header": "heading",
    "text": "text",
    "paragraph": "text",
    "reference": "text",
    "checkbox_selected": "text",
    "checkbox_unselected": "text",
    "list_item": "list_item",
    "table": "table",
    "picture": "picture",
    "chart": "picture",
    "footnote": "footnote",
    "code": "code",
    "formula": "formula",
}

# Lines repeated on many pages (running headers and footers Docling missed)
REPEATED_MIN_PAGES = 3
REPEATED_MIN_SHARE = 0.4
REPEATED_MAX_CHARS = 80

# Tables of contents are near the front and indexes near the back
TOC_ZONE = 0.15
LISTING_MIN_LINES = 5
LISTING_MIN_SHARE = 0.6
CONTENTS_HEADING = re.compile(r"^\s*(table of )?contents\s*$", re.IGNORECASE)
INDEX_HEADING = re.compile(r"^\s*index\s*$", re.IGNORECASE)
# "Chapter 2 Growth ........ 45", "Inflation 12, 45-47", "Foreword  ix"
LISTING_LINE = re.compile(r"(\.{3,}|…|\s)(\d{1,4}(\s*[-–]\s*\d{1,4})?|[ivxlc]{1,7})\s*$", re.IGNORECASE)

TEXT_TYPES = ("heading", "text", "list_item")

# Figures: rendered at 2x for legible chart labels; pictures of these kinds,
# or smaller than FIGURE_MIN_PIXELS, hold no data and are not saved
FIGURE_IMAGE_SCALE = 2.0
FIGURE_MIN_PIXELS = 120
DECORATIVE_FIGURES = {
    "photograph", "natural_image", "icon", "logo", "qr_code", "bar_code", "signature",
    "stamp", "page_thumbnail", "full_page_image", "picture_group", "music", "crossword_puzzle",
}


_parse_lock = threading.Lock()


class ParsingError(Exception):
    """Docling could not parse the PDF."""


# ---------- Cleanup (pure functions, no Docling needed) ----------

def _normalize(text):
    """Lowercase, with a leading or trailing page number replaced.

    'Report 2024 | 17' and 'Report 2024 | 18' become the same line; numbers
    inside the text are kept so different short lines stay different.
    """
    return re.sub(r"^\d+\b|\b\d+$", "#", " ".join(text.lower().split()))


def drop_repeated_lines(elements, page_count):
    """Remove short lines that appear on many pages, such as running headers."""
    if page_count < REPEATED_MIN_PAGES:
        return elements
    pages_by_line = defaultdict(set)
    for element in elements:
        if element["type"] in TEXT_TYPES and len(element["text"]) <= REPEATED_MAX_CHARS:
            pages_by_line[_normalize(element["text"])].add(element["page"])

    threshold = max(REPEATED_MIN_PAGES, REPEATED_MIN_SHARE * page_count)
    repeated = {line for line, pages in pages_by_line.items() if len(pages) >= threshold}
    return [e for e in elements
            if not (e["type"] in TEXT_TYPES and len(e["text"]) <= REPEATED_MAX_CHARS
                    and _normalize(e["text"]) in repeated)]


def _is_listing(page_elements):
    """True if most lines on the page end in a page number, like a contents or index page."""
    lines = [line.strip() for e in page_elements if e["type"] in TEXT_TYPES
             for line in e["text"].splitlines() if line.strip()]
    if len(lines) < LISTING_MIN_LINES:
        return False
    return sum(1 for line in lines if LISTING_LINE.search(line)) / len(lines) >= LISTING_MIN_SHARE


def find_listing_pages(elements, page_count):
    """Return the page numbers of tables of contents and indexes."""
    by_page = defaultdict(list)
    for element in elements:
        by_page[element["page"]].append(element)
    zone = max(1, round(page_count * TOC_ZONE))

    listing = set()
    for page, page_elements in by_page.items():
        headings = [e["text"] for e in page_elements if e["type"] == "heading"]
        in_front, in_back = page <= zone, page > page_count - zone
        if in_front and (any(CONTENTS_HEADING.match(h) for h in headings) or _is_listing(page_elements)):
            listing.add(page)
        elif in_back and (any(INDEX_HEADING.match(h) for h in headings) or _is_listing(page_elements)):
            listing.add(page)
    return listing


def clean_elements(elements, page_count):
    """Remove running headers/footers and contents/index pages.

    Returns (elements, removed) where removed describes what was dropped.
    """
    without_repeats = drop_repeated_lines(elements, page_count)
    listing_pages = find_listing_pages(without_repeats, page_count)
    cleaned = [e for e in without_repeats if e["page"] not in listing_pages]
    return cleaned, {
        "repeated_lines": len(elements) - len(without_repeats),
        "listing_pages": sorted(listing_pages),
    }


# ---------- Turning elements back into page text ----------

def render_element(element):
    """Text of one element as it appears in page text (markdown-like)."""
    text = element["text"]
    if element["type"] == "heading":
        return "#" * min(element["level"] or 1, 6) + " " + text
    if element["type"] == "list_item":
        return text if re.match(r"^\s*([-•*]|\d+[.)])\s", text) else "- " + text
    if element["type"] in ("table", "picture"):
        label = "Table" if element["type"] == "table" else "Figure"
        lines = [f"{label}: {element['caption']}" if element["caption"] else None, text or None]
        return "\n".join(line for line in lines if line)
    return text


def build_pages(elements, filename, page_count, printed_pages=None, removed_pages=()):
    """Rebuild one page dict per page (same shape as pdf_ingestion.extract_pages).

    removed_pages (contents/index pages) are left empty on purpose, so they
    are not reported as needing OCR.
    """
    by_page = defaultdict(list)
    for element in elements:
        rendered = render_element(element)
        if rendered.strip():
            by_page[element["page"]].append(rendered)

    pages = []
    for number in range(1, page_count + 1):
        text = "\n\n".join(by_page.get(number, []))
        pages.append({
            "filename": filename,
            "page_number": number,
            "printed_page": printed_pages[number - 1] if printed_pages else str(number),
            "text": text,
            "char_count": len(text),
            "needs_ocr": len(text) < MIN_CHARS_PER_PAGE and number not in removed_pages,
            "error": None,
        })
    return pages


# ---------- Docling ----------

@lru_cache(maxsize=2)
def _converter(ocr):
    """Build a Docling converter once per OCR setting; loading its models is slow."""
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.accelerator_options.num_threads = os.cpu_count() or 4  # Docling's default is 4
    # Digital PDFs already contain their text; OCR is only for scanned pages
    options.do_ocr = ocr
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    # Real heading levels (1, 1.1, 1.1.1) from bookmarks, numbering and font
    # styles, instead of every heading being level 1
    options.heading_hierarchy_options.enabled = True
    # Figure crops and their kind (bar chart, photo, logo, ...), so charts can
    # be described by Gemini and decorative pictures skipped
    options.generate_picture_images = True
    options.images_scale = FIGURE_IMAGE_SCALE
    options.do_picture_classification = True
    # The pypdfium text backend handles letter spacing better than Docling's
    # default: on the GPO report the default produced "Highl ights"
    pdf_format = PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)
    return DocumentConverter(format_options={InputFormat.PDF: pdf_format})


def _figure_class(item):
    """The most likely kind of picture, e.g. "bar_chart" or "logo", or None."""
    classification = getattr(item.meta, "classification", None) if item.meta else None
    predictions = classification.predictions if classification else []
    if not predictions:
        return None
    return max(predictions, key=lambda p: p.confidence or 0).class_name


def _save_figure(item, doc, image_dir, number):
    """Save a useful figure's image as PNG and return its path, or None to skip it."""
    if image_dir is None or _figure_class(item) in DECORATIVE_FIGURES:
        return None
    image = item.get_image(doc)
    if image is None or min(image.size) < FIGURE_MIN_PIXELS:
        return None  # too small to hold a chart
    path = Path(image_dir) / f"figure_{number:03d}_p{item.prov[0].page_no}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.as_posix()


def elements_from_document(doc, image_dir=None):
    """Turn a DoclingDocument into a list of element dicts, in reading order.

    If image_dir is given, images of charts and diagrams are saved there and
    their path is stored in the element's "image".
    """
    from docling_core.types.doc import PictureItem, TableItem

    # Captions that belong to a table or picture are stored with it, not on their own
    attached_captions = {ref.cref for item in (*doc.tables, *doc.pictures) for ref in item.captions}

    elements = []
    for item, _depth in doc.iterate_items():  # body only: page headers/footers are skipped
        label = item.label.value
        if label == "caption":
            if item.self_ref in attached_captions:
                continue
            element_type = "text"
        else:
            element_type = LABEL_TYPES.get(label)
        if element_type is None or not item.prov:
            continue

        prov = item.prov[0]
        page = doc.pages.get(prov.page_no)
        extra = {}
        bbox = prov.bbox.to_top_left_origin(page.size.height) if page else prov.bbox
        caption = None
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=doc).strip()
            caption = item.caption_text(doc).strip() or None
        elif isinstance(item, PictureItem):
            text = ""
            caption = item.caption_text(doc).strip() or None
            extra = {"figure_class": _figure_class(item),
                     "image": _save_figure(item, doc, image_dir, len(elements) + 1)}
        else:
            text = (item.text or "").strip()
            if not text:
                continue

        level = None
        if element_type == "heading":
            level = 1 if label == "title" else getattr(item, "level", 1) or 1
        elements.append({
            "type": element_type,
            "text": text,
            "page": prov.page_no,
            "level": level,
            "bbox": [round(bbox.l, 1), round(bbox.t, 1), round(bbox.r, 1), round(bbox.b, 1)],
            "caption": caption,
            **extra,
        })
    return elements


def parse_pdf(path, filename, page_count, printed_pages=None, ocr=False, image_dir=None):
    """Parse a PDF with Docling and return its cleaned elements and rebuilt pages.

    image_dir is where chart and diagram images are saved (None: not saved).
    Raises ParsingError if Docling is not installed or fails.
    """
    started = time.perf_counter()
    # One parse at a time: Docling's models use a lot of memory and its
    # converters are not safe to share between threads
    with _parse_lock:
        try:
            converter = _converter(ocr)
        except ImportError as err:
            raise ParsingError("Docling is not installed. Run: pip install -r requirements.txt") from err
        try:
            with warnings.catch_warnings():
                # Docling warns for every box it moves back inside the page; harmless and noisy
                warnings.filterwarnings("ignore", message="Provenance bbox", category=UserWarning)
                doc = converter.convert(str(path)).document
            raw_elements = elements_from_document(doc, image_dir)
        except Exception as err:
            raise ParsingError(f"Docling could not parse the PDF: {err}") from err

    elements, removed = clean_elements(raw_elements, page_count)
    pages = build_pages(elements, filename, page_count, printed_pages,
                        removed_pages=set(removed["listing_pages"]))
    return {
        "elements": elements,
        "pages": pages,
        "cleanup": removed,
        "ocr": ocr,
        "image_dir": Path(image_dir).as_posix() if image_dir else None,
        "seconds": round(time.perf_counter() - started, 1),
    }


def printed_page_labels(reader):
    """Page numbers as printed in the PDF ('i', 'ii', '1', ...), one per page.

    Falls back to '1', '2', ... when the PDF defines no labels.
    """
    count = len(reader.pages)
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []
    if len(labels) != count:
        return [str(n) for n in range(1, count + 1)]
    return [str(label) for label in labels]
