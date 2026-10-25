"""Document profiles: how a document is processed and searched.

Every document gets one profile at upload:
  narrative  - mostly plain text (the GPO annual report)
  structured - long, with chapters and sections (the World Bank report)
  visual     - information in tables, charts and footnotes (the Apple report)

The profile is stored with the document so each one can get its own parsing,
chunking and retrieval. If the uploader doesn't choose one, suggest_profile()
picks one from simple signals in the PDF.
"""

import re
from typing import Literal

ProfileName = Literal["narrative", "structured", "visual"]

PROFILES = {
    "narrative": "Mostly plain text, such as letters, policies and short reports.",
    "structured": "Long documents with chapters and sections, such as manuals and long reports.",
    "visual": "Documents with important tables, charts or footnotes, such as ESG or financial reports.",
}
DEFAULT_PROFILE = "narrative"

# Thresholds for suggest_profile(). The GPO annual report (82 pages, photos
# on 40% of them, 57 bookmarks, 8% numeric words) must come out "narrative",
# so images and bookmarks are not used as signals.
STRUCTURED_MIN_PAGES = 100
VISUAL_MIN_NUMERIC_RATIO = 0.12


def numeric_ratio(pages):
    """Share of words that contain a digit. Tables and charts push it up."""
    words = [word for page in pages for word in page["text"].split()]
    if not words:
        return 0.0
    return sum(1 for word in words if re.search(r"\d", word)) / len(words)


def suggest_profile(pages):
    """Return (profile, reason) for a document with these extracted pages."""
    page_count = len(pages)
    if page_count >= STRUCTURED_MIN_PAGES:
        return "structured", f"The document has {page_count} pages."

    ratio = numeric_ratio(pages)
    if ratio >= VISUAL_MIN_NUMERIC_RATIO:
        return "visual", f"{ratio:.0%} of the words are numbers, which suggests tables and charts."
    return "narrative", "The document is mostly plain text."


# How each profile is chunked, enriched and searched. Steps that a profile
# does not list are off for it.
#   describe_figures    Gemini reads charts and diagrams (data points, trends)
#   summarize_tables    Gemini writes a short summary embedded with each table
#   summarize_sections  Gemini summarizes each chapter and the whole document
#   parent_child        search small chunks, answer from their whole section
#   footnote_chunks     footnotes become their own chunks and are attached to
#                       tables, figures and retrieved text on the same page
#   top_k               chunks given to the chat model by default
#   prompt              extra instructions for the chat model
PROFILE_SETTINGS = {
    "narrative": {
        "top_k": 5,
        "prompt": "",
    },
    "structured": {
        "describe_figures": True,
        "summarize_tables": True,
        "summarize_sections": True,
        "parent_child": True,
        "top_k": 6,
        "prompt": (
            "This is a long document with chapters and sections. Each excerpt header names its "
            "section. Mention which chapter or section information comes from when that helps. "
            "Excerpts of type summary describe whole chapters: use them for broad questions, "
            "and prefer detailed excerpts for specific facts."
        ),
    },
    "visual": {
        "describe_figures": True,
        "summarize_tables": True,
        "footnote_chunks": True,
        "top_k": 8,
        "prompt": (
            "Much of this document's information is in tables, charts and footnotes. Excerpts of "
            "type table contain markdown tables, and excerpts of type figure contain a "
            "description of a chart and the data read from it. Quote numbers exactly as written, "
            "with their units and years. Do not calculate new numbers unless the question asks "
            "for it, and then show the calculation. If a footnote qualifies a number (for "
            "example a baseline year, a method or a scope), mention it."
        ),
    },
}


def profile_setting(profile, name, default=False):
    return PROFILE_SETTINGS.get(profile, PROFILE_SETTINGS[DEFAULT_PROFILE]).get(name, default)
