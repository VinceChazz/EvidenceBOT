"""Retrieval tests: parsing, de-duplication and graceful source failure."""

from __future__ import annotations

from processing.deduplication import deduplicate, titles_match
from processing.document_processor import process_documents
from retrieval.arxiv import ArxivRetriever
from retrieval.crossref import CrossrefRetriever
from retrieval.europe_pmc import EuropePMCRetriever
from retrieval.pubmed import PubMedRetriever, build_pubmed_queries
from retrieval.retrieval_manager import RetrievalManager, load_offline_corpus
from retrieval.wikipedia import WikipediaRetriever
from tests.conftest import FakeHttp, fixture_text

QUESTION = "What role does IL-6 play in rheumatoid arthritis?"


# ---------------------------------------------------------------------------
# Individual source parsers
# ---------------------------------------------------------------------------


def test_pubmed_parses_medline_records(settings, fake_http):
    retriever = PubMedRetriever(settings, http=fake_http)
    documents = retriever.search(QUESTION)

    assert len(documents) == 2
    first = documents[0]
    assert first.source_name == "PubMed"
    assert first.pmid == "26719248"
    assert first.doi == "10.1155/2016/2671924"
    assert first.publication_year == 2016
    assert first.url == "https://pubmed.ncbi.nlm.nih.gov/26719248/"
    assert "Interleukin-6" in first.title
    assert "JAK/STAT3" in first.text_passage
    assert first.authors[0].startswith("Hanow")  # MEDLINE exports "LastName Initials"
    assert first.source_type == "review"
    assert documents[1].source_type == "randomized_controlled_trial"


def test_pubmed_xml_fallback(settings):
    http = FakeHttp(
        routes={
            "esearch.fcgi": fixture_text("pubmed_esearch.json"),
            "efetch.fcgi": fixture_text("pubmed_efetch.xml"),
        }
    )
    retriever = PubMedRetriever(settings, http=http)
    # Force the MEDLINE path to yield nothing so the XML fallback is exercised.
    retriever.fetch_medline = lambda ids: []
    documents = retriever.search(QUESTION)

    assert len(documents) == 2
    assert documents[0].pmid == "26719248"
    assert documents[0].publication_year == 2016
    assert "Th17" in documents[0].text_passage
    assert documents[1].source_type == "randomized_controlled_trial"


def test_pubmed_query_ladder():
    queries = build_pubmed_queries("What role does IL-6 play in rheumatoid arthritis?")
    assert queries[0].startswith("What role does IL-6")
    assert any("il-6" in q.lower() for q in queries[1:])
    assert len(queries) <= 4


def test_europe_pmc_parses_and_flags_preprints(settings, fake_http):
    retriever = EuropePMCRetriever(settings, http=fake_http)
    documents = retriever.search(QUESTION)

    assert len(documents) == 2
    review = documents[0]
    assert review.doi == "10.1155/2016/2671924"
    assert review.pmid == "26719248"
    assert review.has_full_text is True  # open-access full text was attached
    assert "tocilizumab" in review.text_passage
    assert review.provenance == "abstract+full_text"

    preprint = documents[1]
    assert preprint.is_preprint is True
    assert preprint.is_peer_reviewed is False
    assert preprint.source_type == "preprint"


def test_arxiv_filters_out_of_scope_categories(settings, fake_http):
    retriever = ArxivRetriever(settings, http=fake_http)
    documents = retriever.search("deep learning single cell RNA sequencing")

    assert len(documents) == 1  # the econ.GN entry is filtered out
    document = documents[0]
    assert document.source_type == "preprint"
    assert document.is_preprint is True
    assert document.arxiv_id == "2101.01234"
    assert document.url == "https://arxiv.org/abs/2101.01234"
    assert document.publication_year == 2021


def test_crossref_metadata_and_missing_abstract(settings, fake_http):
    retriever = CrossrefRetriever(settings, http=fake_http)
    documents = retriever.search(QUESTION)

    assert len(documents) == 2
    with_abstract = documents[0]
    assert with_abstract.doi == "10.1136/annrheumdis-2018-0001"
    assert with_abstract.journal == "Annals of the Rheumatic Diseases"
    assert with_abstract.provenance == "abstract"

    metadata_only = documents[1]
    assert metadata_only.provenance == "title_only"
    assert metadata_only.note  # flagged as metadata-only


def test_wikipedia_is_background_only(settings, fake_http):
    retriever = WikipediaRetriever(settings, http=fake_http)
    documents = retriever.search("interleukin 6")

    assert documents
    for document in documents:
        assert document.is_background is True
        assert document.source_type == "background"
        assert document.is_scientific_evidence is False
        assert "Background material only" in (document.note or "")


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def test_deduplication_merges_same_publication_across_sources(settings, fake_http):
    pubmed_docs = PubMedRetriever(settings, http=fake_http).search(QUESTION)
    epmc_docs = EuropePMCRetriever(settings, http=fake_http).search(QUESTION)
    crossref_docs = CrossrefRetriever(settings, http=fake_http).search(QUESTION)

    corpus = process_documents(pubmed_docs + epmc_docs + crossref_docs, settings)

    dois = [d.doi for d in corpus.documents if d.doi]
    assert len(dois) == len(set(dois)), "duplicate DOIs survived de-duplication"
    merged = [d for d in corpus.documents if d.doi == "10.1155/2016/2671924"]
    assert len(merged) == 1
    assert "Europe PMC" in merged[0].extra.get("also_found_in", [])
    assert corpus.dedup_report.input_count > corpus.dedup_report.output_count


def test_title_similarity_detection():
    assert titles_match(
        "Interleukin-6 and autoimmunity: a review of IL-6 in rheumatoid arthritis",
        "Interleukin-6 and autoimmunity: a review of the pathogenic role of IL-6 in rheumatoid arthritis",
    )
    assert not titles_match(
        "Deep learning for single-cell RNA sequencing",
        "A macroeconomic model of housing markets",
    )


def test_deduplicate_prefers_peer_reviewed_record(settings):
    from tests.conftest import make_document

    preprint = make_document(
        document_id="p1",
        title="IL-6 blockade in rheumatoid arthritis",
        source_name="arXiv",
        source_type="preprint",
        doi="10.1000/xyz",
        is_preprint=True,
        is_peer_reviewed=False,
    )
    review = make_document(
        document_id="p2",
        title="IL-6 blockade in rheumatoid arthritis",
        source_name="PubMed",
        source_type="review",
        doi="10.1000/XYZ",
        is_peer_reviewed=True,
    )
    unique, report = deduplicate([preprint, review])

    assert len(unique) == 1
    assert unique[0].source_name == "PubMed"
    assert unique[0].is_peer_reviewed is True
    assert unique[0].is_preprint is False
    assert report.duplicates_removed == 1


# ---------------------------------------------------------------------------
# Retrieval manager: concurrency, failures, fallbacks
# ---------------------------------------------------------------------------


def test_manager_aggregates_sources_concurrently(settings, fake_http):
    manager = RetrievalManager(settings, http=fake_http)
    outcome = manager.retrieve(QUESTION)

    assert outcome.total_documents >= 4
    assert not outcome.failures
    assert len(outcome.successes) == 5
    assert outcome.fallback_used is False
    names = {d.source_name for d in outcome.documents}
    assert {"PubMed", "Europe PMC", "arXiv", "Crossref", "Wikipedia"} & names


def test_manager_survives_pubmed_outage(settings, fake_http):
    """TEST 5 — one failing API must not break the search."""

    fake_http.failures = ["eutils.ncbi.nlm.nih.gov"]
    manager = RetrievalManager(settings, http=fake_http)
    outcome = manager.retrieve(QUESTION)

    failed = {r.source for r in outcome.failures}
    assert "pubmed" in failed
    assert "PubMed" not in {d.source_name for d in outcome.documents}
    assert outcome.documents, "other sources must still return documents"
    assert any("PubMed" in warning for warning in outcome.warnings)
    assert outcome.fallback_used is False


def test_manager_returns_failure_result_when_source_raises(settings):
    class Broken:
        def search(self, question, limit=None):
            raise RuntimeError("simulated API crash")

    working = FakeHttp(
        routes={"api.crossref.org": fixture_text("crossref_response.json")}
    )
    manager = RetrievalManager(
        settings,
        http=working,
        retrievers={"pubmed": Broken(), "crossref": CrossrefRetriever(settings, http=working)},
    )
    outcome = manager.retrieve(QUESTION, sources=["pubmed", "crossref"])

    assert [r.source for r in outcome.failures] == ["pubmed"]
    assert outcome.documents
    assert "simulated API crash" in outcome.failures[0].error


def test_offline_fallback_is_flagged(settings):
    all_failing = FakeHttp(routes={}, failures=["http"])
    manager = RetrievalManager(settings.with_overrides(enable_offline_sample_fallback=True),
                               http=all_failing)
    outcome = manager.retrieve(QUESTION)

    assert outcome.fallback_used is True
    assert outcome.documents
    assert all(doc.is_sample for doc in outcome.documents)
    assert any("NOT live results" in w for w in outcome.warnings)


def test_offline_corpus_loader_marks_samples():
    documents = load_offline_corpus()
    assert len(documents) >= 10
    assert all(doc.is_sample for doc in documents)
    assert any(doc.is_background for doc in documents)
    assert any(doc.is_preprint for doc in documents)


def test_empty_question_is_handled(settings, fake_http):
    manager = RetrievalManager(settings, http=fake_http)
    outcome = manager.retrieve("   ")
    assert outcome.documents == []
    assert outcome.warnings
