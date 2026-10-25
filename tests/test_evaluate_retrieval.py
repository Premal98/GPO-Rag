"""Tests for the retrieval evaluation metrics."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_retrieval import score_question, summarize

EXAMPLE_FILE = Path(__file__).parent.parent / "eval" / "retrieval_questions.example.json"


def results(*pages, document_id="doc"):
    return [{"rank": rank, "page_number": page, "document_id": document_id}
            for rank, page in enumerate(pages, start=1)]


def test_relevant_chunk_ranked_first():
    assert score_question(results(6, 2, 9), [6]) == {"hit": True, "recall": 1.0, "reciprocal_rank": 1.0}


def test_relevant_chunk_ranked_third():
    score = score_question(results(1, 2, 6), [6])
    assert score["hit"] is True
    assert score["reciprocal_rank"] == pytest.approx(1 / 3)


def test_miss():
    assert score_question(results(1, 2), [6]) == {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0}


def test_recall_counts_distinct_expected_pages():
    assert score_question(results(5, 5, 9), [5, 6])["recall"] == 0.5


def test_results_from_another_document_do_not_count():
    assert score_question(results(6, document_id="other"), [6], document_id="doc")["hit"] is False


def test_summary_averages_scores():
    summary = summarize([
        {"hit": True, "recall": 1.0, "reciprocal_rank": 1.0},
        {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0},
    ])
    assert summary == {"questions": 2, "hit_at_k": 0.5, "recall_at_k": 0.5, "mrr": 0.5}


def test_example_questions_file_is_valid():
    questions = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8"))
    assert questions
    for item in questions:
        assert item["question"].strip()
        assert item["expected_pages"] and all(isinstance(p, int) for p in item["expected_pages"])


def test_a_chunk_spanning_pages_counts_for_each_of_them():
    section = [{"rank": 1, "page_number": 4, "page_end": 6, "document_id": "doc"}]

    assert score_question(section, [5]) == {"hit": True, "recall": 1.0, "reciprocal_rank": 1.0}
    assert score_question(section, [5, 9])["recall"] == 0.5


@pytest.mark.parametrize("name", ["apple_questions.template.json",
                                  "world_bank_questions.template.json"])
def test_question_templates_are_valid(name):
    questions = json.loads((EXAMPLE_FILE.parent / name).read_text(encoding="utf-8"))

    assert len(questions) >= 10
    for item in questions:
        assert item["question"].strip() and item["type"]
        assert item["expected_pages"] == []  # filled in by whoever has read the PDF
