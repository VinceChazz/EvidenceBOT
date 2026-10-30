"""Citation validation tests, including invalid ids and hallucinated claims."""

from __future__ import annotations

import pytest

from evidence.citation_validator import (
    build_citation_records,
    normalise_citation_placement,
    parse_citation_markers,
    strip_citation_markers,
    validate_citations,
)
from tests.conftest import make_passage


@pytest.fixture()
def evidence():
    return [
        make_passage(
            document_id="doc-1",
            title="Interleukin-6 trans-signalling in rheumatoid arthritis",
            relevance_score=0.81,
            doi="10.1155/2016/2671924",
            pmid="26719248",
            url="https://pubmed.ncbi.nlm.nih.gov/26719248/",
            text=(
                "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium, promotes "
                "Th17 differentiation and drives osteoclast differentiation in rheumatoid arthritis."
            ),
        ),
        make_passage(
            document_id="doc-2",
            title="Tocilizumab randomised trial in rheumatoid arthritis",
            relevance_score=0.72,
            doi="10.1136/annrheumdis-2018-0001",
            url="https://doi.org/10.1136/annrheumdis-2018-0001",
            source_type="randomized_controlled_trial",
            text=(
                "In this randomised controlled trial, IL-6 receptor blockade with tocilizumab "
                "improved ACR response rates and reduced disease activity scores in rheumatoid "
                "arthritis."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Marker parsing
# ---------------------------------------------------------------------------


def test_parse_markers_supports_lists_and_ranges():
    assert parse_citation_markers("Text [1] and [2,3] and [4-6].") == [1, 2, 3, 4, 5, 6]
    assert parse_citation_markers("No citations here.") == []
    assert parse_citation_markers("[1,1,1]") == [1]


def test_trailing_markers_are_moved_before_the_full_stop():
    text = "IL-6 drives synovial inflammation. [1] Tocilizumab improved ACR responses. [2]"
    normalised = normalise_citation_placement(text)
    assert normalised == "IL-6 drives synovial inflammation [1]. Tocilizumab improved ACR responses [2]."


def test_strip_citation_markers_removes_only_invalid_numbers():
    text = "Claim one [1]. Claim two [2]. Claim three [9]. Combined [1,9]."
    cleaned = strip_citation_markers(text, [9])
    assert "[9]" not in cleaned
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "[1]" in cleaned.split("Combined")[1]


# ---------------------------------------------------------------------------
# Valid answers
# ---------------------------------------------------------------------------


def test_valid_answer_passes(settings, evidence):
    answer = (
        "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium and promotes Th17 "
        "differentiation in rheumatoid arthritis [1]. IL-6 receptor blockade with tocilizumab "
        "improved ACR response rates and reduced disease activity scores [2]."
    )
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is True
    assert report.invalid_citations == []
    assert report.unsupported_claims == []
    assert report.uncited_claims == []
    assert report.used_citations == [1, 2]
    assert report.verification_label.startswith("Fully evidence-verified")
    assert report.needs_regeneration() is False


def test_citation_map_matches_document_metadata(settings, evidence):
    report = validate_citations("A claim [1].", evidence, settings)
    record = report.citation_map[1]

    assert record.document_id == "doc-1"
    assert record.doi == "10.1155/2016/2671924"
    assert record.pmid == "26719248"
    assert record.url == "https://pubmed.ncbi.nlm.nih.gov/26719248/"
    assert record.number == 1
    assert report.citation_map[2].document_id == "doc-2"


def test_build_citation_records_numbers_sequentially(evidence):
    records = build_citation_records(evidence)
    assert [r.number for r in records] == [1, 2]
    assert records[0].source_name == "PubMed"


# ---------------------------------------------------------------------------
# Invalid citations (TEST 4)
# ---------------------------------------------------------------------------


def test_invalid_citation_is_rejected_and_removed(settings, evidence):
    """TEST 4 — a citation id that does not exist must never be displayed."""

    answer = "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium [1]. A fabricated claim [7]."
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is False
    assert report.invalid_citations == [7]
    assert 7 not in report.citation_map
    assert "[7]" not in report.sanitized_answer
    assert "[1]" in report.sanitized_answer
    assert report.needs_regeneration() is True
    assert any("never retrieved" in note for note in report.notes)


def test_invented_url_and_doi_are_flagged(settings, evidence):
    answer = (
        "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium [1]. "
        "Details are available at https://fake-journal.example/il6 and doi:10.9999/fake.2020."
    )
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is False
    assert report.invented_urls == ["https://fake-journal.example/il6"]
    assert report.invented_dois
    assert "[link removed]" in report.sanitized_answer
    assert report.needs_regeneration() is True


def test_retrieved_urls_are_allowed(settings, evidence):
    answer = (
        "IL-6 receptor blockade with tocilizumab improved ACR response rates [2]. "
        "The trial is listed at https://doi.org/10.1136/annrheumdis-2018-0001."
    )
    report = validate_citations(answer, evidence, settings)
    assert report.invented_urls == []


# ---------------------------------------------------------------------------
# Unsupported calls (TEST 7)
# ---------------------------------------------------------------------------


def test_unsupported_claim_is_detected(settings, evidence):
    """TEST 7 — a claim the cited passage does not contain must be rejected."""

    answer = (
        "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium [1]. "
        "Vitamin D supplementation eliminates cancer risk in every patient [1]."
    )
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is False
    assert report.unsupported_claims
    flagged = " ".join(c.claim for c in report.unsupported_claims).lower()
    assert "vitamin d" in flagged
    assert report.needs_regeneration() is True


def test_causal_overstatement_is_rejected(settings, evidence):
    answer = "Tocilizumab cures rheumatoid arthritis in most patients [2]."
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is False
    assert "causal" in report.unsupported_claims[0].reason.lower()


def test_fabricated_numeric_value_is_rejected(settings, evidence):
    answer = "Tocilizumab improved ACR response rates by 87 percent [2]."
    report = validate_citations(answer, evidence, settings)

    assert report.is_valid is False
    assert "87" in report.unsupported_claims[0].reason


def test_uncited_substantive_claim_is_reported(settings, evidence):
    answer = (
        "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium [1]. "
        "Rheumatoid arthritis is a chronic autoimmune disease that affects the joints worldwide."
    )
    report = validate_citations(answer, evidence, settings)

    assert report.uncited_claims
    assert report.is_valid is True  # citations themselves are valid...
    assert report.verification_label.startswith("Citations verified")  # ...but not fully verified
    assert report.all_claims_supported is False


def test_trailing_marker_still_counts_as_citation(settings, evidence):
    answer = (
        "Interleukin-6 activates JAK/STAT3 signalling in inflamed synovium and promotes Th17 "
        "differentiation in rheumatoid arthritis. [1]"
    )
    report = validate_citations(answer, evidence, settings)
    assert report.used_citations == [1]
    assert not report.uncited_claims


def test_no_citations_at_all(settings, evidence):
    answer = "Interleukin-6 is involved in rheumatoid arthritis pathogenesis and synovial inflammation."
    report = validate_citations(answer, evidence, settings)
    assert report.used_citations == []
    assert report.uncited_claims
    assert report.all_claims_supported is False
