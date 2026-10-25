"""Measures retrieval quality against a list of questions with known answer pages.

Run from the project folder (needs GOOGLE_API_KEY and SUPABASE_DATABASE_URL):
    python -m scripts.evaluate_retrieval eval/retrieval_questions.example.json
    python -m scripts.evaluate_retrieval eval/apple_questions.template.json --document-id <id>

The questions file is a JSON list like:
    [{"question": "What was GPO's revenue in FY 2018?",
      "expected_pages": [6],
      "document_id": "9e4f9e917d951597",     <- optional; --document-id sets it for all
      "type": "fact"}]                        <- optional; results are also shown per type

A result counts as relevant when its pages (a chunk can span several, e.g.
a whole section) include one of the expected pages, and it comes from the
expected document if one is given. Questions whose expected_pages are still
empty (in the templates) are skipped.

Metrics, averaged over all questions:
  hit@k     share of questions with at least one relevant chunk in the top k
  recall@k  share of each question's expected pages that appear in the top k
  MRR       mean of 1 / rank of the first relevant chunk (0 if none); 1.0
            means the right chunk is always ranked first
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from app.retrieval import RetrievalError, search_chunks


def result_pages(result):
    """All pages a result covers (most chunks are on one page)."""
    first = result["page_number"]
    return set(range(first, (result.get("page_end") or first) + 1))


def relevant_ranks(results, expected_pages, document_id=None):
    """Ranks (1-based) of results that cover an expected page."""
    return [
        r["rank"] for r in results
        if result_pages(r) & set(expected_pages)
        and (document_id is None or r["document_id"] == document_id)
    ]


def score_question(results, expected_pages, document_id=None):
    ranks = relevant_ranks(results, expected_pages, document_id)
    found_pages = set().union(*(result_pages(r) for r in results if r["rank"] in ranks)) \
        & set(expected_pages)
    return {
        "hit": bool(ranks),
        "recall": len(found_pages) / len(set(expected_pages)),
        "reciprocal_rank": 1 / ranks[0] if ranks else 0.0,
    }


def summarize(scores):
    count = len(scores)
    return {
        "questions": count,
        "hit_at_k": sum(s["hit"] for s in scores) / count,
        "recall_at_k": sum(s["recall"] for s in scores) / count,
        "mrr": sum(s["reciprocal_rank"] for s in scores) / count,
    }


def describe(result):
    pages = f"p{result['page_number']}"
    if (result.get("page_end") or result["page_number"]) != result["page_number"]:
        pages += f"-{result['page_end']}"
    kind = result.get("chunk_type") or "text"
    return pages if kind == "text" else f"{pages} {kind}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions_file", type=Path)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Chunks per question (default: the document profile's setting).")
    parser.add_argument("--document-id", default=None,
                        help="Search this document for every question.")
    args = parser.parse_args()

    questions = json.loads(args.questions_file.read_text(encoding="utf-8"))
    scores, by_type, skipped, top_k = [], defaultdict(list), 0, args.top_k
    for item in questions:
        if not item.get("expected_pages"):
            skipped += 1
            continue
        document_id = args.document_id or item.get("document_id")
        try:
            response = search_chunks(item["question"], top_k=args.top_k, document_id=document_id)
        except RetrievalError as err:
            print(f"[FAIL] {err.message}")
            return 1
        top_k = response["top_k"]

        results = response["results"]
        score = score_question(results, item["expected_pages"], document_id)
        scores.append(score)
        by_type[item.get("type", "untyped")].append(score)

        mark = "HIT " if score["hit"] else "MISS"
        print(f"[{mark}] {item['question']}")
        print(f"       expected pages {item['expected_pages']}; got "
              f"{', '.join(describe(r) for r in results) or 'nothing'}")
        for warning in response["warnings"]:
            print(f"       warning: {warning}")

    if skipped:
        print(f"\nSkipped {skipped} question(s) without expected_pages. Open the PDF, find the "
              "page(s) with each answer, and fill them in.")
    if not scores:
        print("No questions to evaluate.")
        return 1
    summary = summarize(scores)
    print(f"\n{summary['questions']} questions, top_k={top_k}")
    print(f"hit@k:    {summary['hit_at_k']:.2f}")
    print(f"recall@k: {summary['recall_at_k']:.2f}")
    print(f"MRR:      {summary['mrr']:.2f}")
    if len(by_type) > 1:
        print("\nBy question type:")
        for kind, kind_scores in sorted(by_type.items()):
            part = summarize(kind_scores)
            print(f"  {kind:<10} {part['questions']:>3} questions   hit@k {part['hit_at_k']:.2f}   "
                  f"MRR {part['mrr']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
