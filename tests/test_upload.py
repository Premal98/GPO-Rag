"""Tests for the PDF upload endpoint (validation and text extraction)."""

import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import config
from app.main import app
from pdf_helpers import make_pdf, page_text


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Save uploads to a temporary folder instead of the real uploads folder."""
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(upload_dir):
    return TestClient(app)


def upload(client, content, filename="report.pdf", content_type="application/pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, content_type)})


def saved_pdfs(upload_dir):
    return list(upload_dir.glob("*.pdf"))


def test_extracts_text_page_by_page(client, upload_dir):
    response = upload(client, make_pdf([page_text(1), page_text(2), page_text(3)]))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["filename"] == "report.pdf"
    assert body["page_count"] == 3
    assert body["pages_needing_ocr"] == []
    for number, page in enumerate(body["pages"], start=1):
        assert page["filename"] == "report.pdf"
        assert page["page_number"] == number
        assert page["text"] == page_text(number)
        assert page["needs_ocr"] is False
    # The file is saved in the uploads folder, named after its document ID
    saved = saved_pdfs(upload_dir)
    assert len(saved) == 1
    assert saved[0].name == f"{body['document_id']}_report.pdf"


def test_handles_80_page_pdf(client):
    response = upload(client, make_pdf([page_text(n) for n in range(1, 81)]))

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 80
    assert [p["page_number"] for p in body["pages"]] == list(range(1, 81))
    assert body["pages"][79]["text"] == page_text(80)


def test_scanned_pdf_reports_ocr_required(client, upload_dir):
    response = upload(client, make_pdf([None, None, None]))

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "ocr_required"
    assert "OCR" in body["message"]
    assert body["pages_needing_ocr"] == [1, 2, 3]
    assert body["chunk_count"] == 0
    # The scanned file is kept so it can be OCR'd later
    assert len(saved_pdfs(upload_dir)) == 1


def test_partly_scanned_pdf_reports_which_pages_need_ocr(client):
    response = upload(client, make_pdf([page_text(1), None, page_text(3)]))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["pages_needing_ocr"] == [2]
    assert body["pages"][1]["needs_ocr"] is True
    assert body["pages"][1]["text"] == ""


def test_rejects_empty_file(client, upload_dir):
    response = upload(client, b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_rejects_non_pdf_extension(client, upload_dir):
    response = upload(client, b"hello", filename="notes.txt", content_type="text/plain")

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_rejects_wrong_content_type(client):
    response = upload(client, make_pdf([page_text(1)]), content_type="image/png")

    assert response.status_code == 400
    assert "content type" in response.json()["detail"]


def test_rejects_text_file_renamed_to_pdf(client, upload_dir):
    response = upload(client, b"I am just a text file pretending to be a PDF.")

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_rejects_corrupted_pdf(client, upload_dir):
    response = upload(client, b"%PDF-1.4\nthis is not real pdf content")

    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_rejects_file_over_size_limit(client, upload_dir, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)  # about 1 KB
    large_pdf = make_pdf([page_text(n) for n in range(1, 30)])

    response = upload(client, large_pdf)

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_unsafe_filename_is_saved_inside_uploads_folder(client, upload_dir):
    response = upload(client, make_pdf([page_text(1)]), filename="../../evil.pdf")

    assert response.status_code == 200
    saved = saved_pdfs(upload_dir)
    assert len(saved) == 1
    assert saved[0].name.endswith("_evil.pdf")
    # Folder parts are not kept in the metadata used for citations either
    body = response.json()
    assert body["filename"] == "evil.pdf"
    assert all(page["filename"] == "evil.pdf" for page in body["pages"])


def test_windows_path_is_reduced_to_the_filename(client):
    response = upload(client, make_pdf([page_text(1)]), filename="C:\\Users\\me\\Annual Report.pdf")

    assert response.status_code == 200
    assert response.json()["filename"] == "Annual Report.pdf"


def test_uppercase_extension_is_accepted(client):
    assert upload(client, make_pdf([page_text(1)]), filename="REPORT.PDF").status_code == 200


def test_missing_content_type_is_accepted_when_the_file_is_a_pdf(client):
    response = client.post("/documents/upload", files={"file": ("report.pdf", make_pdf([page_text(1)]))})

    assert response.status_code == 200


def upload_with_profile(client, content, profile):
    return client.post(f"/documents/upload?profile={profile}",
                       files={"file": ("report.pdf", content, "application/pdf")})


PLAIN_TEXT = "The office served many libraries and readers across the country this year."


def test_profile_is_suggested_when_not_given(client):
    plain = upload(client, make_pdf([PLAIN_TEXT])).json()
    numbers = upload(client, make_pdf(["Year 2019 2020 2021 CO2 25.1 23.4 20.0"])).json()

    assert plain["profile"] == "narrative"
    assert plain["profile_reason"].startswith("Suggested automatically")
    assert numbers["profile"] == "visual"


def test_profile_chosen_at_upload_is_kept(client):
    body = upload_with_profile(client, make_pdf([page_text(1)]), "visual").json()

    assert body["profile"] == "visual"
    assert body["profile_reason"] == "Chosen at upload."


def test_unknown_profile_is_rejected(client, upload_dir):
    response = upload_with_profile(client, make_pdf([page_text(1)]), "fancy")

    assert response.status_code == 422
    assert saved_pdfs(upload_dir) == []


def test_reupload_with_another_profile_rechunks(client):
    pdf = make_pdf([PLAIN_TEXT])
    upload(client, pdf)

    same = upload(client, pdf).json()
    changed = upload_with_profile(client, pdf, "structured").json()
    again = upload(client, pdf).json()  # no profile: keeps the last one

    assert same["duplicate"] is True and same["profile"] == "narrative"
    assert changed["duplicate"] is False and changed["profile"] == "structured"
    assert again["duplicate"] is True and again["profile"] == "structured"


# ---------- Which parser runs (Docling is replaced by a fake, see conftest.py) ----------

def test_narrative_documents_use_plain_text_extraction(client, layout_parser):
    body = upload_with_profile(client, make_pdf([PLAIN_TEXT]), "narrative").json()

    assert body["parser"] == "pypdf"
    assert body["layout_summary"] is None
    assert layout_parser.calls == []


@pytest.mark.parametrize("profile", ["structured", "visual"])
def test_layout_profiles_are_parsed_with_docling(client, layout_parser, profile):
    body = upload_with_profile(client, make_pdf([PLAIN_TEXT, page_text(2)]), profile).json()

    assert body["parser"] == "docling"
    assert body["message"].startswith("Parsed with layout analysis")
    assert body["layout_summary"]["elements"] == 2
    assert body["pages"][0]["text"] == PLAIN_TEXT
    assert layout_parser.calls == [{"ocr": False}]  # digital PDF: no OCR
    # The raw extraction results stay on disk, out of the response
    assert "layout" not in body and "text_pages" not in body


def test_scanned_pdf_is_sent_to_ocr_whatever_the_profile(client, layout_parser):
    body = upload_with_profile(client, make_pdf([None, None]), "narrative")

    assert layout_parser.calls == [{"ocr": True}]
    # The fake OCR finds no text either, so the upload still reports it
    assert body.status_code == 422
    assert body.json()["status"] == "ocr_required"


def test_partly_scanned_narrative_document_skips_ocr(client, layout_parser):
    upload_with_profile(client, make_pdf([PLAIN_TEXT, None]), "narrative")

    assert layout_parser.calls == []


def test_partly_scanned_layout_document_uses_ocr(client, layout_parser):
    upload_with_profile(client, make_pdf([PLAIN_TEXT, None]), "visual")

    assert layout_parser.calls == [{"ocr": True}]


def test_docling_failure_falls_back_to_plain_text(client, layout_parser):
    from app.parsing import ParsingError
    layout_parser.error = ParsingError("Docling could not parse the PDF: boom.")

    response = upload_with_profile(client, make_pdf([PLAIN_TEXT]), "visual")

    assert response.status_code == 200
    body = response.json()
    assert body["parser"] == "pypdf"
    assert body["profile"] == "visual"
    assert "plain text extraction was used instead" in body["warnings"][0]
    assert body["chunk_count"] == 1


def test_switching_profiles_reuses_the_docling_result(client, layout_parser):
    pdf = make_pdf([PLAIN_TEXT])

    upload_with_profile(client, pdf, "visual")
    narrative = upload_with_profile(client, pdf, "narrative").json()
    structured = upload_with_profile(client, pdf, "structured").json()

    assert narrative["parser"] == "pypdf"
    assert structured["parser"] == "docling"
    assert len(layout_parser.calls) == 1


def test_encrypted_pdf_is_rejected(client, upload_dir):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret-password")
    buffer = io.BytesIO()
    writer.write(buffer)

    response = upload(client, buffer.getvalue())

    assert response.status_code == 400
    assert "password-protected" in response.json()["detail"]
    assert saved_pdfs(upload_dir) == []


def test_a_plain_text_fallback_is_not_reused_once_docling_works(client, layout_parser):
    from app.parsing import ParsingError
    pdf = make_pdf([PLAIN_TEXT])
    layout_parser.error = ParsingError("Docling could not parse the PDF: out of memory.")
    first = upload_with_profile(client, pdf, "visual").json()
    layout_parser.error = None

    second = upload_with_profile(client, pdf, "visual").json()

    assert first["parser"] == "pypdf"
    assert second["duplicate"] is False and second["parser"] == "docling"


def test_results_saved_by_older_versions_are_processed_again(client, upload_dir):
    import json
    pdf = make_pdf([PLAIN_TEXT])
    body = upload(client, pdf).json()
    path = upload_dir / "processed" / f"{body['document_id']}.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    del old["chunk_settings"]
    path.write_text(json.dumps(old), encoding="utf-8")

    again = upload(client, pdf)

    assert again.status_code == 200 and again.json()["duplicate"] is False


def test_processed_results_are_written_without_temporary_leftovers(client, upload_dir):
    upload(client, make_pdf([PLAIN_TEXT]))

    assert [p.suffix for p in (upload_dir / "processed").iterdir()] == [".json"]
