"""Loads settings from the .env file so the rest of the app can use them."""

import os

from dotenv import load_dotenv

# Read variables from the .env file into the environment
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SUPABASE_DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL", "")
APP_NAME = os.getenv("APP_NAME", "GPO RAG Chatbot")

# PDF uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))

# Chunking (sizes are in characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be 0 or more and smaller than CHUNK_SIZE.")

# Embeddings. The dimension must match the vector size in
# sql/001_create_document_chunks.sql; ingestion checks this before inserting.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))
# Seconds to wait for one embedding request before giving up on it
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
# After a rate-limit error Gemini says how long to wait (about 60s on the free
# tier); waits longer than this give up instead
EMBEDDING_MAX_RETRY_WAIT_SECONDS = float(os.getenv("EMBEDDING_MAX_RETRY_WAIT_SECONDS", "90"))

# Chat (answer generation)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash")
# Optional second model, used only when CHAT_MODEL is overloaded (HTTP 503)
CHAT_FALLBACK_MODEL = os.getenv("CHAT_FALLBACK_MODEL", "").strip()
# Leave empty to use the model's default; Google advises against changing it for Gemini 3+
_chat_temperature = os.getenv("CHAT_TEMPERATURE", "").strip()
CHAT_TEMPERATURE = float(_chat_temperature) if _chat_temperature else None
# Includes the model's "thinking" tokens on Gemini 2.5+, so keep it generous
CHAT_MAX_OUTPUT_TOKENS = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "4096"))
# Seconds to wait for one Gemini request, and how many times to retry it
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "60"))
CHAT_MAX_RETRIES = int(os.getenv("CHAT_MAX_RETRIES", "2"))
CHAT_MAX_QUESTION_CHARS = int(os.getenv("CHAT_MAX_QUESTION_CHARS", "2000"))

if EMBEDDING_TIMEOUT_SECONDS <= 0 or CHAT_TIMEOUT_SECONDS <= 0:
    raise ValueError("EMBEDDING_TIMEOUT_SECONDS and CHAT_TIMEOUT_SECONDS must be greater than 0.")
if CHAT_TEMPERATURE is not None and not 0 <= CHAT_TEMPERATURE <= 2:
    raise ValueError("CHAT_TEMPERATURE must be between 0 and 2.")
if CHAT_MAX_RETRIES < 0 or CHAT_MAX_OUTPUT_TOKENS < 1 or CHAT_MAX_QUESTION_CHARS < 1:
    raise ValueError("CHAT_MAX_RETRIES must be 0 or more; CHAT_MAX_OUTPUT_TOKENS and "
                     "CHAT_MAX_QUESTION_CHARS must be at least 1.")

# Enrichment: while a structured or visual document is processed, Gemini
# describes its charts, summarizes its tables and (structured) its chapters.
# Requests are batched and spaced out to stay within free-tier rate limits;
# the MAX_ settings cap how many items one document may use.
ENRICHMENT_ENABLED = os.getenv("ENRICHMENT_ENABLED", "true").strip().lower() in ("1", "true", "yes")
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL", "").strip() or CHAT_MODEL
ENRICHMENT_MAX_FIGURES = int(os.getenv("ENRICHMENT_MAX_FIGURES", "40"))
ENRICHMENT_MAX_TABLES = int(os.getenv("ENRICHMENT_MAX_TABLES", "80"))
ENRICHMENT_MAX_SECTIONS = int(os.getenv("ENRICHMENT_MAX_SECTIONS", "30"))
# Seconds between two Gemini requests (4 = at most 15 requests per minute)
ENRICHMENT_MIN_INTERVAL_SECONDS = float(os.getenv("ENRICHMENT_MIN_INTERVAL_SECONDS", "4"))
ENRICHMENT_TIMEOUT_SECONDS = float(os.getenv("ENRICHMENT_TIMEOUT_SECONDS", "120"))

if min(ENRICHMENT_MAX_FIGURES, ENRICHMENT_MAX_TABLES, ENRICHMENT_MAX_SECTIONS) < 0         or ENRICHMENT_MIN_INTERVAL_SECONDS < 0 or ENRICHMENT_TIMEOUT_SECONDS <= 0:
    raise ValueError("ENRICHMENT_MAX_* and ENRICHMENT_MIN_INTERVAL_SECONDS must be 0 or more, "
                     "and ENRICHMENT_TIMEOUT_SECONDS greater than 0.")

# Retrieval. TOP_K is the default number of chunks returned; callers can ask
# for up to RETRIEVAL_MAX_TOP_K. MIN_SIMILARITY (cosine, -1 to 1) drops weak
# matches; leave it empty to always return the top K.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_MAX_TOP_K = int(os.getenv("RETRIEVAL_MAX_TOP_K", "50"))
_min_similarity = os.getenv("RETRIEVAL_MIN_SIMILARITY", "").strip()
RETRIEVAL_MIN_SIMILARITY = float(_min_similarity) if _min_similarity else None

# Hybrid search: this many candidates come from vector search and as many
# from keyword search; they are merged, reranked, and trimmed to top_k chunks
# or RETRIEVAL_CONTEXT_MAX_CHARS of text, whichever comes first.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "30"))
RETRIEVAL_CONTEXT_MAX_CHARS = int(os.getenv("RETRIEVAL_CONTEXT_MAX_CHARS", "12000"))
# Local reranker (a cross-encoder that reads the question and each candidate
# together). Downloaded once from Hugging Face; runs on the CPU.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base").strip()
# Only the best this-many merged candidates are reranked (about 2.5 s per 30 on a laptop CPU)
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

if RETRIEVAL_CANDIDATES < 1 or RERANK_CANDIDATES < 1 or RETRIEVAL_CONTEXT_MAX_CHARS < 1000:
    raise ValueError("RETRIEVAL_CANDIDATES and RERANK_CANDIDATES must be at least 1, and "
                     "RETRIEVAL_CONTEXT_MAX_CHARS at least 1000.")
if not 1 <= RETRIEVAL_TOP_K <= RETRIEVAL_MAX_TOP_K:
    raise ValueError("RETRIEVAL_TOP_K must be between 1 and RETRIEVAL_MAX_TOP_K.")
if RETRIEVAL_MIN_SIMILARITY is not None and not -1 <= RETRIEVAL_MIN_SIMILARITY <= 1:
    raise ValueError("RETRIEVAL_MIN_SIMILARITY must be between -1 and 1.")

if not 128 <= EMBEDDING_DIMENSION <= 3072:
    raise ValueError("EMBEDDING_DIMENSION must be between 128 and 3072 for Gemini embeddings.")
if not 1 <= EMBEDDING_BATCH_SIZE <= 100:
    raise ValueError("EMBEDDING_BATCH_SIZE must be between 1 and 100 (Gemini's per-request limit).")
