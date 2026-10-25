"""Stand-ins for Gemini, Supabase and Docling, so tests run offline, fast and free."""

import math
import re

from app.parsing import build_pages
from app.pdf_ingestion import extract_pages, open_pdf


class FakeLayoutParser:
    """Behaves like app.parsing.parse_pdf(): one text element per page with text.

    Loading real Docling takes seconds and downloads models, so unit tests
    use this. Set error to make it raise, as a broken parse would.
    """

    def __init__(self, error=None):
        self.error = error
        self.calls = []  # {"ocr": ...} for each call

    def __call__(self, path, filename, page_count, printed_pages=None, ocr=False, image_dir=None):
        self.calls.append({"ocr": ocr})
        if self.error:
            raise self.error
        elements = [
            {"type": "text", "text": page["text"], "page": page["page_number"], "level": None,
             "bbox": None, "caption": None}
            for page in extract_pages(open_pdf(path), filename) if page["text"]
        ]
        return {
            "elements": elements,
            "pages": build_pages(elements, filename, page_count, printed_pages),
            "cleanup": {"repeated_lines": 0, "listing_pages": []},
            "ocr": ocr,
            "image_dir": str(image_dir) if image_dir else None,
            "seconds": 0.0,
        }


class FakeEnricher:
    """Behaves like app.enrichment.enrich_layout() without calling Gemini.

    Set summary to give every table that summary, or warnings to return them.
    """

    def __init__(self, summary=None, warnings=()):
        self.summary = summary
        self.warnings = list(warnings)
        self.calls = []  # profile of each call

    def __call__(self, layout, profile, filename, llm=None, sleep=None, on_status=None):
        self.calls.append(profile)
        if self.summary:
            for element in layout["elements"]:
                if element["type"] == "table":
                    element["summary"] = self.summary
        return list(self.warnings)


class FakeEmbedder:
    """Behaves like GoogleGenerativeAIEmbeddings.embed_documents() and embed_query()."""

    def __init__(self, dimension, failures=None, query_vector=None):
        self.dimension = dimension
        self.failures = list(failures or [])  # exceptions to raise on the first calls
        self.query_vector = query_vector  # what embed_query() returns, if set
        self.calls = []  # the list of texts sent in each call

    def _vector(self, text):
        # Simple, non-normalized vectors that differ per text
        return [float(len(text))] + [1.0] * (self.dimension - 1)

    def embed_documents(self, texts, batch_size=100):
        self.calls.append(list(texts))
        if self.failures:
            raise self.failures.pop(0)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.calls.append([text])
        if self.failures:
            raise self.failures.pop(0)
        if self.query_vector is not None:
            return list(self.query_vector)
        return self._vector(text)


def cosine_distance(a, b):
    """What pgvector's <=> operator computes."""
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# A few of the words Postgres' english text search ignores
STOPWORDS = {"a", "an", "and", "are", "did", "do", "does", "for", "how", "in", "is", "it", "of",
             "on", "or", "the", "to", "was", "were", "what", "when", "which", "who", "why"}


class FakeChunkStore:
    """Behaves like app.vector_store.ChunkStore, but keeps rows in memory."""

    def __init__(self, dimension, error=None, migrated=True):
        self.dimension = dimension
        self.error = error  # exception to raise from every method, to simulate an outage
        self.migrated = migrated  # False simulates a database without sql/002
        self.rows = {}
        self.documents = {}  # document_id -> registry row
        self.status_history = []  # (document_id, status) for each status change
        self.searches = []  # (query_vector, limit, document_id) for each vector search
        self.keyword_searches = []  # (terms, section_pattern) for each keyword search

    def _check(self):
        if self.error:
            raise self.error

    def status(self, document_id=None):
        self._check()
        migrated = self.dimension is not None and self.migrated
        return {
            "embedding_dimension": self.dimension,
            "migrated": migrated,
            "document": self.documents.get(document_id) if migrated else None,
        }

    def register(self, document_id, filename="Annual-Report.pdf", profile="narrative"):
        """Add a registry row directly, as if the document had been ingested."""
        self.documents[document_id] = {
            "document_id": document_id, "filename": filename, "profile": profile,
            "status": "ready", "status_message": None, "page_count": None,
            "chunk_count": self._count(document_id), "created_at": None, "ingested_at": None,
        }

    def _count(self, document_id):
        return sum(1 for r in self.rows.values() if r["document_id"] == document_id)

    def count_chunks(self, document_id):
        self._check()
        return self._count(document_id)

    def save_chunks(self, document, rows, replace=False):
        self._check()
        document_id = document["document_id"]
        existing = self._count(document_id)
        if existing and not replace:
            return {"skipped": True, "inserted": 0, "existing": existing}
        if replace:
            self.rows = {k: r for k, r in self.rows.items() if r["document_id"] != document_id}
        inserted = 0
        for row in rows:
            if row["chunk_id"] not in self.rows:
                self.rows[row["chunk_id"]] = row
                inserted += 1
        self.register(document_id, document["filename"], document["profile"])
        self.documents[document_id]["page_count"] = document["page_count"]
        return {"skipped": False, "inserted": inserted, "existing": existing}

    def set_document_status(self, document_id, filename, status, message=None, profile=None):
        self._check()
        if document_id not in self.documents:
            self.register(document_id, filename, profile or "narrative")
        row = self.documents[document_id]
        row.update({"filename": filename, "status": status, "status_message": message})
        if profile:
            row["profile"] = profile
        self.status_history.append((document_id, status))

    def fail_unfinished(self, message):
        self._check()
        busy = [d for d in self.documents.values()
                if d["status"] in ("uploaded", "parsing", "enriching", "embedding")]
        for row in busy:
            row.update({"status": "failed", "status_message": message})
        return len(busy)

    def list_documents(self):
        self._check()
        return list(self.documents.values())

    def delete_document(self, document_id):
        self._check()
        chunks = self._count(document_id)
        self.rows = {k: r for k, r in self.rows.items() if r["document_id"] != document_id}
        return {"document_deleted": self.documents.pop(document_id, None) is not None,
                "chunks_deleted": chunks}

    def _embedded(self, document_id):
        return [r for r in self.rows.values()
                if r.get("embedding") is not None
                and (document_id is None or r["document_id"] == document_id)]

    def count_embedded_chunks(self, document_id=None):
        self._check()
        return len(self._embedded(document_id))

    @staticmethod
    def _columns(r):
        return {
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "content": r["content"],
            "source_filename": r["source_filename"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            "metadata": r["metadata"],
            "chunk_type": r.get("chunk_type", "text"),
            "section_path": r.get("section_path"),
            "page_end": r.get("page_end"),
            "printed_page": r.get("printed_page"),
            "parent_chunk_id": r.get("parent_chunk_id"),
        }

    def _searchable(self, document_id, section_pattern):
        rows = self._embedded(document_id)
        if section_pattern is not None:  # the patterns used are valid Python regexes too
            rows = [r for r in rows
                    if re.search(section_pattern, r.get("section_path") or "\0", re.IGNORECASE)]
        return rows

    def search(self, query_vector, limit, document_id=None, section_pattern=None):
        self._check()
        self.searches.append((list(query_vector), limit, document_id))
        scored = [{**self._columns(r), "distance": cosine_distance(query_vector, r["embedding"])}
                  for r in self._searchable(document_id, section_pattern)]
        return sorted(scored, key=lambda r: r["distance"])[:limit]

    def keyword_search(self, terms, query_vector, limit, document_id=None, section_pattern=None):
        """Rows containing any of the terms, those with more matching terms first.

        Like Postgres' english configuration, common words never match.
        """
        self._check()
        self.keyword_searches.append((list(terms), section_pattern))
        terms = [t for t in terms if t not in STOPWORDS]
        matched = []
        for r in self._searchable(document_id, section_pattern):
            words = set(re.findall(r"[a-z0-9]+", r["content"].lower()))
            hits = sum(1 for term in terms if term in words)
            if hits:
                matched.append((hits, {**self._columns(r),
                                       "distance": cosine_distance(query_vector, r["embedding"])}))
        return [row for _, row in sorted(matched, key=lambda m: -m[0])][:limit]

    def search_candidates(self, query_vector, terms, limit, document_id=None, section_pattern=None):
        return (self.search(query_vector, limit, document_id, section_pattern),
                self.keyword_search(terms, query_vector, limit, document_id, section_pattern))

    def get_chunks(self, chunk_ids):
        self._check()
        return {cid: self._columns(self.rows[cid]) for cid in chunk_ids if cid in self.rows}

    def get_footnotes(self, document_id, pages):
        self._check()
        footnotes = {}
        for r in sorted(self.rows.values(), key=lambda r: r["chunk_index"]):
            if r["document_id"] == document_id and r.get("chunk_type") == "footnote" \
                    and r["page_number"] in pages:
                footnotes.setdefault(r["page_number"], []).append(r["content"])
        return footnotes
