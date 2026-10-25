"""Stores processing results as JSON files in uploads/processed/.

This is a simple local stand-in until the database (Supabase) is added.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

# Document IDs are 16 lowercase hex characters (see save_upload)
DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")


def _result_path(upload_dir, document_id):
    return Path(upload_dir) / "processed" / f"{document_id}.json"


def save_result(upload_dir, result):
    path = _result_path(upload_dir, result["document_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a temporary file and swap it in, so a reader (e.g. a status poll
    # while a background job saves) never sees a half-written file
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                # Windows refuses while a reader has the file open for a moment
                if attempt == 4:
                    raise
                time.sleep(0.2)
    finally:
        temp.unlink(missing_ok=True)


def load_result(upload_dir, document_id):
    """Return the stored result for a document, or None if there isn't one."""
    if not DOCUMENT_ID_PATTERN.match(document_id):
        return None  # also blocks paths like "../secret"
    path = _result_path(upload_dir, document_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
