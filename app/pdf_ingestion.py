"""Saves uploaded PDFs to disk and extracts their text page by page."""

import hashlib
import re
import uuid
from pathlib import Path

from pypdf import PdfReader

# Every real PDF file starts with these bytes
PDF_MAGIC = b"%PDF-"

# Content types browsers and tools commonly send for PDF files
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}

# A page with fewer characters than this is probably an image (scanned) page
MIN_CHARS_PER_PAGE = 20

# Read uploads 1 MB at a time so large files don't fill memory
CHUNK_SIZE = 1024 * 1024


class PDFIngestionError(Exception):
    """A problem with the uploaded file that we report back to the user."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def safe_filename(name):
    """Remove folder parts and unusual characters from a filename."""
    name = Path(name).name  # "../../evil.pdf" -> "evil.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.pdf"


def display_filename(name):
    """The filename without folder parts, as shown in chunk metadata and citations.

    Unlike safe_filename() it keeps spaces and other readable characters.
    Handles both / and \\ so Windows paths are cleaned on any OS.
    """
    return re.split(r"[\\/]", name)[-1].strip() or "upload.pdf"


def validate_upload(filename, content_type):
    """Check the filename and content type before reading the file."""
    if not filename.lower().endswith(".pdf"):
        raise PDFIngestionError("Only .pdf files are allowed.")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise PDFIngestionError(f"Unsupported content type '{content_type}'. Please upload a PDF.")


def save_upload(file_obj, filename, upload_dir, max_bytes):
    """Write the upload to disk, checking its size and that it really is a PDF.

    The file is named after a hash of its contents, so uploading the same PDF
    twice is detected and only one copy is kept.

    Returns (saved_path, document_id). If anything is wrong, the partly
    written file is deleted and a PDFIngestionError is raised.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f".{uuid.uuid4().hex}.part"
    hasher = hashlib.sha256()

    size = 0
    try:
        with open(temp_path, "wb") as out:
            while chunk := file_obj.read(CHUNK_SIZE):
                if size == 0 and PDF_MAGIC not in chunk[:1024]:
                    raise PDFIngestionError("The file is not a valid PDF (missing PDF header).")
                size += len(chunk)
                if size > max_bytes:
                    max_mb = max_bytes / (1024 * 1024)
                    raise PDFIngestionError(f"File is too large. The limit is {max_mb:g} MB.", 413)
                hasher.update(chunk)
                out.write(chunk)
        if size == 0:
            raise PDFIngestionError("The uploaded file is empty.")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    # The same file contents always give the same document ID
    document_id = hasher.hexdigest()[:16]

    existing = next(upload_dir.glob(f"{document_id}_*.pdf"), None)
    if existing:
        temp_path.unlink()
        return existing, document_id

    saved_path = upload_dir / f"{document_id}_{safe_filename(filename)}"
    temp_path.replace(saved_path)
    return saved_path, document_id


def open_pdf(path):
    """Open the PDF with pypdf, turning library errors into friendly messages."""
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise PDFIngestionError("Could not read the PDF. The file may be corrupted.") from exc

    if reader.is_encrypted:
        try:
            # Some PDFs are "encrypted" with an empty password and still open fine
            decrypted = reader.decrypt("")
        except Exception:
            decrypted = False
        if not decrypted:
            raise PDFIngestionError("The PDF is password-protected. Please upload an unlocked copy.")

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PDFIngestionError("Could not read the PDF pages. The file may be corrupted.") from exc

    if page_count == 0:
        raise PDFIngestionError("The PDF has no pages.")

    return reader


def extract_pages(reader, filename):
    """Extract text from every page, keeping the filename and page number."""
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        error = None
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            text = ""
            error = f"Text extraction failed on this page: {exc}"

        pages.append({
            "filename": filename,
            "page_number": page_number,
            "text": text,
            "char_count": len(text),
            "needs_ocr": len(text) < MIN_CHARS_PER_PAGE,
            "error": error,
        })
    return pages


def summarize_extraction(pages):
    """Return (status, message, pages_needing_ocr) describing how extraction went.

    status is one of:
      - "success":      every page has text
      - "partial":      some pages have no text and need OCR
      - "ocr_required": no page has text (likely a scanned PDF)
    """
    ocr_pages = [p["page_number"] for p in pages if p["needs_ocr"]]
    if not ocr_pages:
        return "success", f"Extracted text from all {len(pages)} pages.", ocr_pages
    if len(ocr_pages) == len(pages):
        message = ("No extractable text was found. This PDF appears to be scanned "
                   "or image-only and needs OCR before it can be used.")
        return "ocr_required", message, ocr_pages
    message = (f"Extracted text from {len(pages) - len(ocr_pages)} of {len(pages)} pages. "
               f"{len(ocr_pages)} page(s) have little or no text and may need OCR.")
    return "partial", message, ocr_pages
