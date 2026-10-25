"""Checks that the Gemini API key, embedding model, and chat model work.

Makes two small API calls (one embedding, one chat reply). Does not touch
the database. Run from the project folder:
    python -m scripts.check_gemini
"""

import sys

from app import config
from app.chat import ChatError, build_messages, create_chat_model, message_text, parse_model_reply
from app.embeddings import (
    EmbeddingError,
    create_embedder,
    embed_query,
    format_query_for_embedding,
    is_embedding_configured,
)

SAMPLE_CHUNK = {"filename": "sample.pdf", "page_number": 1,
                "content": "The sky in this sample document is green."}


def main():
    if not is_embedding_configured():
        print("[SKIP] GOOGLE_API_KEY is not set (or is still a placeholder) in .env.")
        print("       Nothing was called. Get a key at https://aistudio.google.com/apikey")
        return 1

    print(f"Embedding with {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIMENSION} dims) ...")
    try:
        vector = embed_query(create_embedder(),
                             format_query_for_embedding("hello", config.EMBEDDING_MODEL),
                             expected_dimension=config.EMBEDDING_DIMENSION, max_retries=1)
    except EmbeddingError as err:
        print(f"[FAIL] {err}")
        print("       Check GOOGLE_API_KEY, EMBEDDING_MODEL and EMBEDDING_DIMENSION in .env.")
        return 1
    print(f"[OK]   Got a {len(vector)}-dimension embedding.")

    print(f"Asking {config.CHAT_MODEL} a question about a sample excerpt ...")
    try:
        message = create_chat_model().invoke(
            build_messages("What colour is the sky in the document?", [SAMPLE_CHUNK]))
        reply = parse_model_reply(message_text(message))
    except ChatError as err:
        print(f"[FAIL] {err.message}")
        return 1
    except Exception as err:
        print(f"[FAIL] The chat model request failed: {err}")
        print("       Check CHAT_MODEL in .env against https://ai.google.dev/gemini-api/docs/models")
        return 1
    print(f"[OK]   Answer: {reply['answer']}")
    print(f"       supported={reply['supported']}, sources={reply['source_ids']}")
    print("Gemini is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
