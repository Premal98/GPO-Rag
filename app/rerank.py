"""Reorders search candidates with a local cross-encoder reranker.

Vector search compares the question and each chunk separately (two
embeddings). A cross-encoder reads the question and a chunk together, so it
judges relevance much better, but is too slow to run on every chunk; it is
used on the ~30-60 candidates from hybrid search only.

The default model, BAAI/bge-reranker-base (MIT licence), is downloaded from
Hugging Face on first use (about 1.1 GB) and runs on the CPU.
"""

import logging
import threading
from functools import lru_cache

from app import config

logger = logging.getLogger(__name__)

MAX_TOKENS = 512
_lock = threading.Lock()  # one prediction at a time; the model is not thread-safe


@lru_cache(maxsize=1)
def _load_model(name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, max_length=MAX_TOKENS, device="cpu")


class CrossEncoderReranker:
    def __init__(self, model_name):
        self.model_name = model_name

    def score(self, query, texts):
        """Relevance score for each text (higher is more relevant)."""
        with _lock:
            model = _load_model(self.model_name)
            return [float(s) for s in model.predict([(query, t) for t in texts], batch_size=16)]


def get_reranker():
    """The configured reranker, or None when RERANK_ENABLED is false."""
    return CrossEncoderReranker(config.RERANK_MODEL) if config.RERANK_ENABLED else None


def warm_up():
    """Load the model in advance, so the first question isn't slowed by it."""
    reranker = get_reranker()
    if reranker is None:
        return
    try:
        reranker.score("warm up", ["warm up"])
    except Exception:
        logger.warning("Could not load the reranker model %s; search will work without it.",
                       config.RERANK_MODEL, exc_info=True)
