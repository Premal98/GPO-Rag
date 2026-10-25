"""Runs the full upload flow: validate -> save -> extract -> chunk -> store."""

from collections import Counter
from pathlib import Path

from app.chunkers import chunk_layout
from app.chunking import chunk_pages, chunk_to_dict
from app.document_store import load_result, save_result
from app.enrichment import enrich_layout
from app.parsing import ParsingError, parse_pdf, printed_page_labels
from app.pdf_ingestion import (
    display_filename,
    extract_pages,
    open_pdf,
    save_upload,
    summarize_extraction,
    validate_upload,
)
from app.profiles import DEFAULT_PROFILE, suggest_profile

# Profiles whose documents are parsed with Docling (layout-aware). Narrative
# documents use pypdf's plain text, which is much faster and enough for them.
LAYOUT_PROFILES = {"structured", "visual"}


def _layout_summary(layout):
    return {
        "elements": len(layout["elements"]),
        "by_type": dict(Counter(e["type"] for e in layout["elements"])),
        "removed": layout["cleanup"],
        "ocr": layout["ocr"],
        "seconds": layout["seconds"],
    }


def save_pdf_upload(file_obj, filename, content_type, upload_dir, max_bytes):
    """Validate and save an upload. Returns (saved_path, document_id, filename).

    This part is fast, so it runs inside the request even when the rest of
    the processing happens in the background.
    """
    validate_upload(filename, content_type)
    filename = display_filename(filename)
    saved_path, document_id = save_upload(file_obj, filename, upload_dir, max_bytes)
    return saved_path, document_id, filename


def process_pdf_upload(file_obj, filename, content_type, upload_dir, max_bytes,
                       chunk_size, chunk_overlap, profile=None):
    """Save an uploaded PDF, process it, and return the result dictionary."""
    saved_path, document_id, filename = save_pdf_upload(file_obj, filename, content_type,
                                                        upload_dir, max_bytes)
    return process_saved_pdf(saved_path, document_id, filename, upload_dir,
                             chunk_size, chunk_overlap, profile=profile)


def process_saved_pdf(saved_path, document_id, filename, upload_dir, chunk_size, chunk_overlap,
                      profile=None, reuse_result=True, on_status=None):
    """Extract, parse and chunk a saved PDF, and store the result on disk.

    profile is "narrative", "structured" or "visual". If it is None, the
    profile of an earlier upload of the same file is kept, or one is
    suggested from the PDF. Structured and visual documents, and scanned
    PDFs of any profile, are parsed with Docling; the rest use pypdf.

    If the same file was processed before with the same profile and chunk
    settings (and reuse_result is True), the stored result is returned
    instead of doing the work again. Otherwise earlier extraction results are
    reused where possible. on_status(stage) is called when a slow stage starts.
    """
    on_status = on_status or (lambda stage: None)
    chunk_settings = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

    previous = load_result(upload_dir, document_id)
    if previous:
        # Results saved before profiles existed are narrative
        previous.setdefault("profile", DEFAULT_PROFILE)
        previous.setdefault("profile_reason", "Processed before profiles were added.")
    # A result is only reused if it was complete: with chunks, and for a layout
    # profile parsed by Docling (not the plain-text fallback after a Docling failure)
    reusable = previous and previous.get("chunks") and (
        previous["profile"] not in LAYOUT_PROFILES or previous.get("parser") == "docling")
    if reuse_result and reusable and profile in (None, previous["profile"]) \
            and previous.get("chunk_settings") == chunk_settings:
        previous["duplicate"] = True
        previous["message"] = "This file was already processed. " + previous["message"]
        return previous

    if previous:
        # Same file, new profile or chunk settings: reuse what was extracted before
        text_pages = previous.get("text_pages", previous["pages"])
        layout = previous.get("layout")
        filename = previous["filename"]
    else:
        try:
            reader = open_pdf(saved_path)
            text_pages = extract_pages(reader, filename)
        except Exception:
            # Don't keep files we couldn't read
            saved_path.unlink(missing_ok=True)
            raise
        for page, label in zip(text_pages, printed_page_labels(reader)):
            page["printed_page"] = label
        layout = None

    if profile is not None:
        profile_reason = "Chosen at upload."
    elif previous:
        profile, profile_reason = previous["profile"], previous["profile_reason"]
    else:
        profile, profile_reason = suggest_profile(text_pages)
        profile_reason = "Suggested automatically: " + profile_reason

    status, message, ocr_pages = summarize_extraction(text_pages)
    pages, parser, warnings = text_pages, "pypdf", []

    # Docling for layout-aware profiles, and to OCR scanned PDFs of any profile
    structure_aware = profile in LAYOUT_PROFILES
    if structure_aware or status == "ocr_required":
        ocr = status != "success"
        # Figure images are only needed (and saved) for structured and visual documents
        image_dir = Path(upload_dir) / "figures" / document_id if structure_aware else None
        if layout is None or (ocr and not layout["ocr"]) \
                or (image_dir and not layout.get("image_dir")):
            on_status("parsing")
            try:
                layout = parse_pdf(saved_path, filename, len(text_pages),
                                   [p.get("printed_page") or str(p["page_number"]) for p in text_pages],
                                   ocr=ocr, image_dir=image_dir)
            except ParsingError as err:
                layout = None
                warnings.append(f"{err} The plain text extraction was used instead.")
        if layout is not None:
            pages, parser = layout["pages"], "docling"
            status, message, ocr_pages = summarize_extraction(pages)
            message = "Parsed with layout analysis (Docling). " + message

    if parser == "docling" and structure_aware:
        warnings += enrich_layout(layout, profile, filename, on_status=on_status)
        chunks = chunk_layout(layout, document_id, filename, profile, chunk_size, chunk_overlap)
    else:
        chunks = [chunk_to_dict(c) for c in chunk_pages(pages, document_id, chunk_size, chunk_overlap)]

    result = {
        "document_id": document_id,
        "status": status,
        "message": message,
        "duplicate": False,
        "filename": filename,
        "profile": profile,
        "profile_reason": profile_reason,
        "parser": parser,
        "layout_summary": _layout_summary(layout) if parser == "docling" else None,
        "warnings": warnings,
        "stored_path": saved_path.as_posix(),
        "page_count": len(pages),
        "pages_needing_ocr": ocr_pages,
        "chunk_settings": chunk_settings,
        "chunk_count": len(chunks),
        "pages": pages,
        "chunks": chunks,
        # Kept so another profile can reuse them without parsing again
        "text_pages": text_pages,
        "layout": layout,
    }
    save_result(upload_dir, result)
    return result
