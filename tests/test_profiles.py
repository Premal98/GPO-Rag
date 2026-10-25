"""Tests for choosing a document profile automatically."""

from app.profiles import numeric_ratio, suggest_profile


def pages(*texts):
    return [{"text": text} for text in texts]


def test_plain_text_is_narrative():
    profile, reason = suggest_profile(pages("The office served many libraries this year."))

    assert profile == "narrative"
    assert "plain text" in reason


def test_many_pages_is_structured():
    profile, reason = suggest_profile(pages(*["Some text."] * 150))

    assert profile == "structured"
    assert "150 pages" in reason


def test_medium_length_document_is_not_structured():
    # The GPO annual report has 82 pages
    assert suggest_profile(pages(*["Some text."] * 82))[0] == "narrative"


def test_number_heavy_text_is_visual():
    table_like = "Year 2019 2020 2021 Emissions 25.1 23.4 20.0 Energy 12% 14% 18%"

    profile, reason = suggest_profile(pages(table_like))

    assert profile == "visual"
    assert "numbers" in reason


def test_annual_report_with_some_figures_stays_narrative():
    # About 8% numeric words, like the GPO annual report
    text = " ".join(["word"] * 92 + ["2018"] * 8)

    assert suggest_profile(pages(text))[0] == "narrative"


def test_numeric_ratio_handles_empty_pages():
    assert numeric_ratio(pages("", "  ")) == 0.0
    assert numeric_ratio(pages("a 1 b 2")) == 0.5
