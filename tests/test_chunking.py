"""Tests for chunking: the splitter itself, the chunks endpoint, and duplicate handling."""

import pytest
from fastapi.testclient import TestClient

from app import config, pipeline
from app.chunking import chunk_pages
from app.main import app
from pdf_helpers import long_text, make_pdf, page_text


def make_page(number, text, filename="guide.pdf"):
    return {"filename": filename, "page_number": number, "text": text}


# ---------- The chunking function on its own ----------

def test_long_page_is_split_into_chunks_within_size():
    pages = [make_page(1, long_text(600))]  # about 4,000 characters

    chunks = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_chunks_overlap():
    pages = [make_page(1, long_text(600))]

    chunks = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)

    # The start of each chunk repeats text from the end of the one before it
    for previous, current in zip(chunks, chunks[1:]):
        first_words = current.page_content[:30]
        assert first_words in previous.page_content


def test_metadata_has_source_page_and_chunk_id():
    pages = [make_page(1, long_text(300)), make_page(2, long_text(300, prefix="other"))]

    chunks = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "guide.pdf"
        assert chunk.metadata["page"] in (1, 2)
        assert chunk.metadata["document_id"] == "abc123"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_id"] == chunk.id
        assert chunk.id.startswith(f"abc123_p{chunk.metadata['page']}_c")
    # Chunk IDs are unique
    assert len({c.id for c in chunks}) == len(chunks)
    # Page 2's chunks only contain page 2's text
    page_2 = [c for c in chunks if c.metadata["page"] == 2]
    assert page_2 and all("word" not in c.page_content for c in page_2)


def test_empty_and_blank_pages_produce_no_chunks():
    pages = [make_page(1, ""), make_page(2, "   \n\n  "), make_page(3, page_text(3))]

    chunks = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 3
    assert all(c.page_content.strip() for c in chunks)


def test_chunk_ids_are_the_same_every_time():
    pages = [make_page(1, long_text(600))]

    first = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)
    second = chunk_pages(pages, "abc123", chunk_size=1000, chunk_overlap=150)

    assert [c.id for c in first] == [c.id for c in second]


# ---------- Through the API ----------

@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(upload_dir):
    return TestClient(app)


def upload(client, content, filename="guide.pdf"):
    return client.post("/documents/upload", files={"file": (filename, content, "application/pdf")})


def test_upload_creates_chunks_that_can_be_inspected(client):
    pdf = make_pdf([long_text(400), page_text(2), None])
    body = upload(client, pdf).json()

    assert body["chunk_count"] > 1
    assert body["chunk_settings"] == {"chunk_size": 1000, "chunk_overlap": 150}

    response = client.get(body["chunks_url"])
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == body["chunk_count"]
    for chunk in data["chunks"]:
        assert chunk["metadata"]["source"] == "guide.pdf"
        assert chunk["metadata"]["page"] in (1, 2)  # page 3 is blank
        assert chunk["chunk_id"] == chunk["metadata"]["chunk_id"]


def test_chunks_can_be_filtered_by_page(client):
    body = upload(client, make_pdf([long_text(400), page_text(2)])).json()

    data = client.get(f"{body['chunks_url']}?page=2").json()

    assert data["chunk_count"] == 1
    assert data["chunks"][0]["text"] == page_text(2)


def test_unknown_document_returns_404(client):
    assert client.get("/documents/0000000000000000/chunks").status_code == 404
    assert client.get("/documents/not-a-real-id/chunks").status_code == 404


def test_duplicate_upload_is_not_processed_again(client, upload_dir, monkeypatch):
    pdf = make_pdf([long_text(400), page_text(2)])
    first = upload(client, pdf).json()

    # If extraction runs again, this test fails
    def fail(*args, **kwargs):
        raise AssertionError("PDF was extracted twice")
    monkeypatch.setattr(pipeline, "extract_pages", fail)

    second = upload(client, pdf, filename="copy-of-guide.pdf").json()

    assert second["duplicate"] is True
    assert second["document_id"] == first["document_id"]
    assert second["chunk_count"] == first["chunk_count"]
    assert len(list(upload_dir.glob("*.pdf"))) == 1


def test_new_chunk_settings_rechunk_without_extracting_again(client, monkeypatch):
    pdf = make_pdf([long_text(400)])
    first = upload(client, pdf).json()

    def fail(*args, **kwargs):
        raise AssertionError("PDF was extracted twice")
    monkeypatch.setattr(pipeline, "extract_pages", fail)
    monkeypatch.setattr(config, "CHUNK_SIZE", 500)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 50)

    second = upload(client, pdf).json()

    assert second["duplicate"] is False
    assert second["chunk_settings"] == {"chunk_size": 500, "chunk_overlap": 50}
    assert second["chunk_count"] > first["chunk_count"]
