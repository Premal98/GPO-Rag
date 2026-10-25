"""Splits extracted PDF pages into smaller, overlapping chunks.

Each page is split on its own, so every chunk belongs to exactly one page
and keeps an accurate page number for citations.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_splitter(chunk_size, chunk_overlap):
    """Build a splitter that prefers to break at paragraphs, then lines, then words."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # records where each chunk starts in the page text
    )


def chunk_pages(pages, document_id, chunk_size, chunk_overlap):
    """Turn a list of extracted pages into LangChain Documents.

    Each chunk's metadata has: source (filename), page, chunk_id,
    chunk_index (position in the whole document), document_id, start_index,
    and printed_page when the printed page label differs from the page number.
    Pages without text produce no chunks, and blank chunks are skipped.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    chunks = []

    for page in pages:
        if not page["text"].strip():
            continue

        metadata = {"source": page["filename"], "page": page["page_number"]}
        if page.get("printed_page") and page["printed_page"] != str(page["page_number"]):
            metadata["printed_page"] = page["printed_page"]  # e.g. "xii", for citations
        page_docs = splitter.create_documents([page["text"]], metadatas=[metadata])
        page_docs = [doc for doc in page_docs if doc.page_content.strip()]

        for index_on_page, doc in enumerate(page_docs):
            chunk_id = f"{document_id}_p{page['page_number']}_c{index_on_page}"
            doc.id = chunk_id
            doc.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": len(chunks),
                "document_id": document_id,
            })
            chunks.append(doc)

    return chunks


def chunk_to_dict(doc):
    """Convert a chunk to plain data for JSON responses and storage."""
    return {
        "chunk_id": doc.id,
        "text": doc.page_content,
        "char_count": len(doc.page_content),
        "metadata": doc.metadata,
    }


def dict_to_chunk(data):
    """Turn stored chunk data back into a LangChain Document."""
    return Document(id=data["chunk_id"], page_content=data["text"], metadata=data["metadata"])
