"""Entry point for the FastAPI application."""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.chat import ChatError, answer_question
from app.document_registry import (
    DocumentRegistryError,
    delete_document,
    get_document,
    list_documents,
)
from app.document_store import load_result
from app.ingestion import IngestionError, ingest_document
from app.jobs import is_active, recover_interrupted, submit_document
from app.pdf_ingestion import PDFIngestionError
from app.pipeline import process_saved_pdf, save_pdf_upload
from app.profiles import ProfileName
from app.rerank import warm_up
from app.retrieval import RetrievalError, search_chunks


@asynccontextmanager
async def lifespan(app):
    # Jobs don't survive a restart; don't leave their documents stuck as "parsing"
    recover_interrupted()
    # Load the reranker model in the background so the first question isn't slow
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title=config.APP_NAME, lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, examples=["What was GPO's revenue in FY 2018?"])
    top_k: int | None = Field(default=None, ge=1, le=config.RETRIEVAL_MAX_TOP_K,
                              description=f"Defaults to {config.RETRIEVAL_TOP_K}.")
    document_id: str | None = Field(default=None, description="Search only this document.")
    min_similarity: float | None = Field(default=None, ge=-1, le=1,
                                         description="Hide chunks below this cosine similarity.")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject misspelled fields instead of ignoring them

    question: str = Field(min_length=1, max_length=config.CHAT_MAX_QUESTION_CHARS,
                          examples=["What was GPO's revenue in FY 2018?"])
    top_k: int | None = Field(default=None, ge=1, le=config.RETRIEVAL_MAX_TOP_K,
                              description="Chunks to use as context. Defaults to the document "
                                          "profile's setting (narrative 5, structured 6, visual 8).")
    document_id: str = Field(min_length=1, max_length=64, examples=["9e4f9e917d951597"],
                             description="The document to answer from. GET /documents lists them.")
    min_similarity: float | None = Field(default=None, ge=-1, le=1,
                                         description="Ignore chunks below this cosine similarity.")


class ChatSource(BaseModel):
    source_id: str
    filename: str
    page_number: int
    page_end: int
    printed_page: str | None = None
    document_id: str
    chunk_id: str
    chunk_type: str
    section_path: str | None
    similarity: float
    excerpt: str


class ChatCitation(BaseModel):
    filename: str
    pages: list[int]


class ChatRetrieval(BaseModel):
    status: str
    message: str
    chunks_retrieved: int
    top_k: int
    min_similarity: float | None
    document_id: str | None
    profile: str | None = None
    reranked: bool = False
    section_filter: str | None = None


class ChatUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None


class ChatResponse(BaseModel):
    status: Literal["answered", "not_supported", "no_context"]
    question: str
    answer: str
    supported: bool
    sources: list[ChatSource]
    citations: list[ChatCitation]
    model: str
    retrieval: ChatRetrieval
    warnings: list[str]
    timing_ms: dict[str, float]
    usage: ChatUsage | None = None


def _ensure_not_processing(document_id):
    """The step-by-step endpoints must not touch a document a background job is working on."""
    if is_active(document_id):
        raise HTTPException(status_code=409, detail=(
            f"Document '{document_id}' is being processed in the background. Wait until "
            f"GET /documents/{document_id} shows ready or failed."))


@app.get("/health")
def health():
    """Simple check to confirm the API is running."""
    return {"status": "ok"}


@app.get("/documents")
def get_documents():
    """List the ingested documents. Pass a document_id to POST /chat to chat with one."""
    try:
        return list_documents()
    except DocumentRegistryError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)


@app.post("/documents", status_code=202)
def create_document(response: Response, file: UploadFile = File(...),
                    profile: ProfileName | None = None, force: bool = False):
    """Upload a PDF and make it chat-ready in the background (the recommended way).

    Returns at once with the document_id. Parsing, enrichment and embedding
    then run in the background; GET /documents/{document_id} shows the
    status (uploaded, parsing, enriching, embedding, ready or failed).
    Uploading a file that is already ready does nothing unless force=true
    or a different profile is given.
    """
    try:
        result = submit_document(file.file, file.filename or "", file.content_type,
                                 profile=profile, force=force)
    except PDFIngestionError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    except DocumentRegistryError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    if not result["accepted"]:
        response.status_code = 200
    return result


@app.get("/documents/{document_id}")
def get_document_status(document_id: str):
    """One document's status, profile, and what its processing found."""
    try:
        return get_document(document_id, Path(config.UPLOAD_DIR))
    except DocumentRegistryError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)


@app.post("/documents/upload")
def upload_pdf(file: UploadFile = File(...), profile: ProfileName | None = None):
    """Upload a PDF, extract its text page by page, and split it into chunks.

    profile says how the document is processed. If it is left empty, one is
    suggested from the PDF (see profile_reason in the response). Uploading
    the same file again with another profile re-chunks it; then call
    /ingest?force=true to update the stored chunks.
    """
    upload_dir = Path(config.UPLOAD_DIR)
    try:
        saved_path, document_id, filename = save_pdf_upload(
            file.file, file.filename or "", file.content_type, upload_dir,
            int(config.MAX_UPLOAD_MB * 1024 * 1024))
        _ensure_not_processing(document_id)
        result = process_saved_pdf(saved_path, document_id, filename, upload_dir,
                                   config.CHUNK_SIZE, config.CHUNK_OVERLAP, profile=profile)
    except PDFIngestionError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)

    # Chunks can be long, so they are viewed with the endpoint below instead.
    # The raw extraction results are only kept on disk for re-processing.
    response = {key: value for key, value in result.items()
                if key not in ("chunks", "text_pages", "layout")}
    response["chunks_url"] = f"/documents/{result['document_id']}/chunks"

    # A scanned PDF was saved, but we could not get any text from it
    if result["status"] == "ocr_required":
        return JSONResponse(status_code=422, content=response)
    return response


@app.get("/documents/{document_id}/chunks")
def get_chunks(document_id: str, page: int | None = None):
    """Show the chunks for an uploaded document. Use ?page=5 to see one page."""
    result = load_result(Path(config.UPLOAD_DIR), document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = result["chunks"]
    if page is not None:
        chunks = [c for c in chunks if c["metadata"]["page"] == page]

    return {
        "document_id": document_id,
        "filename": result["filename"],
        "chunk_settings": result["chunk_settings"],
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@app.post("/documents/{document_id}/ingest")
def ingest(document_id: str, force: bool = False):
    """Create embeddings for an uploaded document's chunks and store them in Supabase.

    If the document is already stored, nothing happens unless force=true,
    which replaces the stored chunks.
    """
    _ensure_not_processing(document_id)
    try:
        return ingest_document(document_id, Path(config.UPLOAD_DIR), force=force)
    except IngestionError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)


@app.delete("/documents/{document_id}")
def remove_document(document_id: str):
    """Delete a document: its stored chunks, its entry in /documents, and its uploaded files."""
    _ensure_not_processing(document_id)
    try:
        return delete_document(document_id, Path(config.UPLOAD_DIR))
    except DocumentRegistryError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)


@app.post("/search")
def search(request: SearchRequest):
    """Find the stored chunks most similar to a question, without generating an answer.

    Use this to check what the chatbot would retrieve for a question.
    """
    try:
        return search_chunks(
            request.query,
            top_k=request.top_k,
            document_id=request.document_id,
            min_similarity=request.min_similarity,
        )
    except RetrievalError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Answer a question from one ingested document, citing filenames and page numbers.

    If nothing relevant is found, the model is not called and status is "no_context".
    An unknown document_id returns 404.
    """
    try:
        return answer_question(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            min_similarity=request.min_similarity,
        )
    except ChatError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
