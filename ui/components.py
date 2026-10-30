"""Rendering helpers for the EvidenceBot dashboard."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from evidence.evidence_strength import EvidenceStrength
from pipeline import EvidenceBotResult
from utils.helpers import format_seconds, truncate

SOURCE_LABELS = {
    "pubmed": "PubMed / NCBI",
    "europe_pmc": "Europe PMC",
    "arxiv": "arXiv",
    "crossref": "Crossref",
    "wikipedia": "Wikipedia (background)",
    "offline_sample": "Offline sample corpus",
}


def stretch_kwargs() -> Dict[str, Any]:
    """Return the right "full width" argument for the installed Streamlit.

    Newer Streamlit deprecates ``use_container_width`` in favour of
    ``width="stretch"``; older versions only understand the former.
    """

    try:
        import streamlit as st

        version = tuple(int(part) for part in re.findall(r"\d+", st.__version__)[:2])
    except Exception:  # pragma: no cover - defensive
        version = (0, 0)
    if version >= (1, 49):
        return {"width": "stretch"}
    return {"use_container_width": True}


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def section(number: str, title: str) -> None:
    st.markdown(
        f"""<div class="eb-section">
              <span class="eb-section__num">{_esc(number)}</span>
              <span class="eb-section__title">{_esc(title)}</span>
              <span class="eb-section__rule"></span>
            </div>""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="eb-hero">
          <div class="eb-hero__eyebrow">AI4BIOMED · Scientific evidence verification</div>
          <div class="eb-hero__title">Evidence<span>Bot</span></div>
          <div class="eb-hero__subtitle">
            PubMed-grounded answers that cite their sources and abstain when evidence is missing.
            EvidenceBot retrieves real scientific literature from multiple databases, ranks the
            passages, decides whether the retrieved evidence is actually sufficient, and only then
            writes a citation-checked answer. If the evidence does not support the question it says
            so instead of guessing.
          </div>
          <div class="eb-hero__chips">
            <span class="eb-chip">PubMed / NCBI</span>
            <span class="eb-chip">Europe PMC</span>
            <span class="eb-chip">arXiv</span>
            <span class="eb-chip">Crossref</span>
            <span class="eb-chip">Wikipedia (background only)</span>
            <span class="eb-chip eb-chip--accent">No evidence · no answer</span>
          </div>
          <div class="eb-hero__motto">From real literature &nbsp;→&nbsp; to <b>trusted answers</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_outcome_banner(result: EvidenceBotResult) -> None:
    """Green / orange / purple status card reflecting the evidence decision."""

    if result.is_abstention:
        css, icon, label = "abstain", "⛔", "Unanswerable — abstained"
        body = (
            "EvidenceBot did not generate an answer because the retrieved passages do not "
            "contain reliable supporting evidence. No language model was called."
        )
        message = (
            f'<div class="eb-abstain-message">{_esc(result.abstention_message)}</div>'
        )
        reason = f'<div class="eb-small">Reason: {_esc(result.assessment.reason)}</div>'
    elif result.is_partial:
        css, icon, label = "partial", "⚠️", "Partially answerable"
        body = (
            "Part of the question is supported by the retrieved literature; the remaining "
            "parts are reported as evidence gaps."
        )
        message, reason = "", ""
    else:
        css, icon, label = "answerable", "✅", "Answerable — grounded in retrieved evidence"
        body = (
            "The evidence gate accepted the retrieved passages as sufficient. The answer below "
            "may only use those passages and is citation-checked against them."
        )
        message, reason = "", ""

    st.markdown(
        f"""<div class="eb-status eb-status--{css}">
              <div class="eb-status__head">
                <span class="eb-status__icon">{icon}</span>
                <span class="eb-status__label">{label}</span>
              </div>
              <div class="eb-status__body">{body}</div>
              {message}
              {reason}
            </div>""",
        unsafe_allow_html=True,
    )


def render_notices(result: EvidenceBotResult) -> None:
    """Warnings, source failures and verification status."""

    if result.fallback_used:
        st.markdown(
            '<div class="eb-notice eb-notice--danger"><b>Offline sample corpus in use.</b> '
            "All live scientific databases were unreachable from this environment, so the "
            "illustrative sample corpus was loaded to exercise the pipeline. These records are "
            "<b>not real publications</b> and must not be read as scientific findings. "
            "Re-run with network access for live retrieval.</div>",
            unsafe_allow_html=True,
        )

    failures = result.source_failures
    if failures:
        detail = " · ".join(
            f"<b>{_esc(SOURCE_LABELS.get(f['source'], f['source']))}</b>: "
            f"{_esc(truncate(str(f.get('error') or ''), 130))}"
            for f in failures
        )
        st.markdown(
            '<div class="eb-notice eb-notice--warn"><b>Some sources were unavailable.</b> '
            f"The search continued with the remaining databases. {detail}</div>",
            unsafe_allow_html=True,
        )

    if result.validation is not None:
        validation = result.validation
        if validation.is_valid and not validation.uncited_claims:
            css, text = "ok", "Citation validation passed: every citation maps to a retrieved source."
        elif validation.is_valid:
            css, text = "info", (
                "Citation validation passed for every citation, but "
                f"{len(validation.uncited_claims)} statement(s) could not be cross-checked "
                "individually."
            )
        else:
            css, text = "danger", (
                "Citation validation flagged this response: "
                f"{len(validation.invalid_citations)} invalid id(s), "
                f"{len(validation.unsupported_claims)} unsupported claim(s). "
                "Invalid citations were removed — the answer is not presented as verified."
            )
        st.markdown(
            f'<div class="eb-notice eb-notice--{css}">{_esc(text)}</div>',
            unsafe_allow_html=True,
        )

    for warning in result.warnings:
        if "unavailable" in warning.lower() or "offline sample" in warning.lower():
            continue
        st.markdown(
            f'<div class="eb-notice eb-notice--info">{_esc(warning)}</div>',
            unsafe_allow_html=True,
        )


def render_metrics(result: EvidenceBotResult) -> None:
    analytics = result.analytics()
    tiles = [
        ("Documents retrieved", analytics["documents_retrieved"], "unique publications after de-duplication"),
        ("Relevant passages", analytics["relevant_passages"], "above the relevance threshold"),
        ("Used as evidence", analytics["evidence_passages_used"], "passages the answer may cite"),
        ("Citations", analytics["citations_used"], "verified source cards"),
        ("Sources accessed", len(analytics["sources_accessed"]), ", ".join(analytics["sources_accessed"]) or "none"),
        ("Retrieval time", format_seconds(analytics["retrieval_seconds"]), "concurrent multi-source search"),
        ("Top relevance", f"{analytics['top_relevance_score']:.2f}", "hybrid score (not a probability)"),
        ("Evidence strength", result.strength.label.replace(" evidence", ""), f"grade: {result.strength.grade}"),
    ]
    cards = "".join(
        f"""<div class="eb-metric">
              <div class="eb-metric__label">{_esc(label)}</div>
              <div class="eb-metric__value">{_esc(value)}</div>
              <div class="eb-metric__hint">{_esc(hint)}</div>
            </div>"""
        for label, value, hint in tiles
    )
    st.markdown(f'<div class="eb-metrics">{cards}</div>', unsafe_allow_html=True)


def render_answer(result: EvidenceBotResult) -> None:
    """Answer body, claims, limitations and gaps."""

    if result.answer:
        st.markdown(f'<div class="eb-answer">{_markdown_to_html(result.answer)}</div>',
                    unsafe_allow_html=True)
    elif result.extractive_digest:
        st.markdown(
            '<div class="eb-notice eb-notice--info"><b>Extractive evidence digest.</b> '
            "No language model was used. The sentences below are quoted verbatim from the "
            "retrieved passages, with their citation numbers.</div>",
            unsafe_allow_html=True,
        )
        for item in result.extractive_digest:
            st.markdown(
                f"""<div class="eb-quote">
                      <div class="eb-quote__label">Verbatim evidence · [{item['citation']}]</div>
                      “{_esc(item['quote'])}”
                      <div class="eb-small" style="margin-top:6px">{_esc(item['title'])}
                      — {_esc(item['source'])}{(' · ' + str(item['year'])) if item.get('year') else ''}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    if result.claims:
        with st.expander("Supporting claims and their source ids", expanded=False):
            rows = [
                {"Claim": claim["claim"], "Source ids": ", ".join(str(i) for i in claim["source_ids"]) or "—"}
                for claim in result.claims
            ]
            st.dataframe(pd.DataFrame(rows), **stretch_kwargs(), hide_index=True)

    if result.conflicts:
        st.markdown(
            '<div class="eb-notice eb-notice--warn"><b>Conflicting findings reported.</b> '
            + "<br>".join(_esc(c) for c in result.conflicts)
            + "</div>",
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Limitations of the retrieved literature", expanded=False):
            if result.limitations:
                for item in result.limitations:
                    st.markdown(f"- {item}")
            else:
                st.caption("No limitations were returned by the generator.")
    with col_b:
        with st.expander("Unanswered parts of the question", expanded=False):
            if result.unanswered_questions:
                for item in result.unanswered_questions:
                    st.markdown(f"- {item}")
            else:
                st.caption("Nothing was left unanswered from the retrieved evidence.")

    if result.evidence_assessment_text:
        st.markdown(
            f'<div class="eb-small" style="margin-top:8px"><b>Generator evidence note:</b> '
            f"{_esc(result.evidence_assessment_text)}</div>",
            unsafe_allow_html=True,
        )


def _markdown_to_html(text: str) -> str:
    """Minimal, safe Markdown -> HTML for the answer card (bold, italics, bullets)."""

    import re

    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", escaped)
    lines = [line for line in escaped.split("\n")]
    html_parts: List[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "• ")):
            if not in_list:
                html_parts.append("<ul style='margin:6px 0 10px 18px'>")
                in_list = True
            html_parts.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            if stripped:
                html_parts.append(f"<p>{stripped}</p>")
    if in_list:
        html_parts.append("</ul>")
    return "".join(html_parts) or escaped


def render_strength(strength: EvidenceStrength) -> None:
    pct = max(0.0, min(1.0, float(strength.score))) * 100
    st.markdown(
        f"""<div class="eb-strength">
              <div class="eb-strength__label" style="color:{strength.color}">{_esc(strength.label)}</div>
              <div class="eb-strength__score">Rule-based composite index: {strength.score:.2f} / 1.00
                — not a probability, and not produced by an LLM</div>
              <div class="eb-bar"><div class="eb-bar__fill"
                   style="width:{pct:.0f}%; background:linear-gradient(90deg,{strength.color},{strength.color}aa)"></div></div>
              <div class="eb-small">{_esc(" · ".join(strength.reasons))}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("How this grade was computed", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Component": c.name,
                        "Score (0-1)": round(float(c.score), 3),
                        "Weight": round(float(c.weight), 2),
                        "Weighted": round(float(c.score * c.weight), 3),
                        "Basis": str(c.detail),
                    }
                    for c in strength.components
                ]
            ),
            **stretch_kwargs(),
            hide_index=True,
        )
        st.caption(
            "Grades: Strong (composite ≥ 0.72, ≥ 3 studies, consistent), Moderate (≥ 0.52), "
            "Limited (< 0.52 or thin evidence), Conflicting (studies disagree), "
            "Insufficient (evidence gate abstained). Study design, study count, source quality, "
            "consistency, directness and full-text availability are the only inputs."
        )
        for caution in strength.cautions:
            st.markdown(f"- {caution}")


def render_citations(result: EvidenceBotResult) -> None:
    """Expandable citation cards, numbered exactly as used in the answer."""

    if not result.citations:
        st.markdown(
            '<div class="eb-notice eb-notice--info">No sources were used as evidence for this '
            "question, so no citation cards are shown.</div>",
            unsafe_allow_html=True,
        )
        return

    invalid = set(result.validation.invalid_citations) if result.validation else set()

    for record in result.citations:
        header = f"[{record.number}] {truncate(record.title, 96)} — {record.source_name}"
        with st.expander(header, expanded=record.number <= 2):
            badges = [f'<span class="eb-badge">{_esc(record.source_type.replace("_", " "))}</span>']
            badges.append(f'<span class="eb-badge eb-badge--info">{_esc(record.source_name)}</span>')
            if record.is_preprint:
                badges.append('<span class="eb-badge eb-badge--warn">Preprint — not peer reviewed</span>')
            if record.is_sample:
                badges.append('<span class="eb-badge eb-badge--danger">Offline sample record</span>')
            if record.number in invalid:
                badges.append('<span class="eb-badge eb-badge--danger">Invalid citation</span>')
            st.markdown(" ".join(badges), unsafe_allow_html=True)

            authors = ", ".join(record.authors[:6]) + (" et al." if len(record.authors) > 6 else "")
            meta_rows = [
                ("Title", record.title),
                ("Source", record.source_name),
                ("Authors", authors or "not reported"),
                ("Year", record.publication_year or "not reported"),
                ("Journal", record.journal or "not reported"),
                ("DOI", record.doi or "—"),
                ("PMID", record.pmid or "—"),
                ("Passage type", record.provenance),
                ("Relevance", f"{record.relevance_score:.3f} (retrieval score)"),
            ]
            st.markdown(
                '<dl class="eb-kv">'
                + "".join(
                    f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in meta_rows
                )
                + "</dl>",
                unsafe_allow_html=True,
            )
            if record.url:
                st.markdown(f"[Open publication ↗]({record.url})")
            elif record.is_sample:
                st.caption("No link — this is an illustrative offline sample record.")
            st.markdown(
                f"""<div class="eb-quote">
                      <div class="eb-quote__label">Supporting passage used for this citation</div>
                      {_esc(record.passage_text)}
                    </div>""",
                unsafe_allow_html=True,
            )


def render_retrieval_details(result: EvidenceBotResult) -> None:
    """Retrieved passages with scores, plus per-source status."""

    with st.expander("Ranked passages and similarity scores", expanded=False):
        rows = result.ranking.to_rows()
        if rows:
            frame = pd.DataFrame(rows).rename(
                columns={
                    "rank": "#",
                    "passage_id": "Passage",
                    "title": "Publication",
                    "source": "Source",
                    "source_type": "Type",
                    "year": "Year",
                    "dense": "Dense (cosine)",
                    "bm25": "BM25 (0-1)",
                    "lexical": "Concept coverage",
                    "relevance": "Hybrid relevance",
                    "used_as_evidence": "Used as evidence",
                }
            )
            st.dataframe(frame, **stretch_kwargs(), hide_index=True)
            st.caption(
                "Scores are retrieval measurements only (hybrid of dense cosine similarity, BM25 "
                "and concept coverage). A high score does not mean a passage scientifically "
                "supports a claim — that decision belongs to the evidence gate."
            )
        else:
            st.caption("No passages were ranked.")

    with st.expander("All retrieved passages (expandable)", expanded=False):
        if not result.corpus.passages:
            st.caption("No passages retrieved.")
        for passage in result.corpus.passages[:25]:
            label = f"{passage.source_name} · {truncate(passage.title or passage.document_id, 80)}"
            with st.expander(label, expanded=False):
                st.caption(
                    f"type: {passage.source_type} · year: {passage.publication_year or 'n.d.'} · "
                    f"relevance: {passage.relevance_score:.3f} · chunk {passage.chunk_index + 1}"
                )
                st.write(passage.text)

    with st.expander("Source status for this search", expanded=False):
        rows = []
        for source_result in result.retrieval.source_results:
            rows.append(
                {
                    "Source": SOURCE_LABELS.get(source_result.source, source_result.source),
                    "Status": source_result.status,
                    "Documents": source_result.count,
                    "Seconds": round(source_result.elapsed_seconds, 2),
                    "Error": truncate(source_result.error or "", 120),
                }
            )
        st.dataframe(pd.DataFrame(rows), **stretch_kwargs(), hide_index=True)
        if result.fallback_used:
            st.caption(
                "The offline sample corpus is used only when every live source fails; "
                "it is not live retrieval."
            )


def render_evidence_assessment(result: EvidenceBotResult) -> None:
    """Transparent evidence-gate criteria."""

    assessment = result.assessment
    with st.expander("Evidence gate — the four evaluation dimensions", expanded=False):
        st.markdown(
            f"**Decision:** `{assessment.decision}` — {assessment.reason}\n\n"
            f"**Consistency:** {assessment.consistency.label}"
            + (
                " (" + ", ".join(f"{k}: {v}" for k, v in assessment.consistency.stance_counts.items()) + ")"
                if assessment.consistency.stance_counts
                else ""
            )
        )
        rows = []
        for row in assessment.criteria_frame_rows():
            rows.append(
                {
                    "Criterion": str(row["criterion"]),
                    "Value": ", ".join(map(str, row["value"]))
                    if isinstance(row["value"], (list, tuple, set))
                    else str(row["value"]),
                    "Threshold": ", ".join(map(str, row["threshold"]))
                    if isinstance(row["threshold"], (list, tuple, set))
                    else str(row["threshold"]),
                    "Passed": "yes" if row["passed"] else "no",
                    "Explanation": str(row["explanation"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), **stretch_kwargs(), hide_index=True)
        if assessment.aspects:
            st.markdown("**Question coverage by component**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Component": a.component,
                            "Coverage": round(a.coverage, 2),
                            "Covered concepts": ", ".join(a.covered) or "—",
                            "Missing concepts": ", ".join(a.missing) or "—",
                        }
                        for a in assessment.aspects
                    ]
                ),
                **stretch_kwargs(),
                hide_index=True,
            )
        for note in assessment.notes:
            st.markdown(f"- {note}")


def render_comparison(result: EvidenceBotResult) -> None:
    """Side-by-side comparison of findings across publications."""

    if not result.comparison:
        st.caption("No comparable publications were retrieved for this question.")
        return
    frame = pd.DataFrame(result.comparison).rename(
        columns={
            "publication": "Publication",
            "source": "Source",
            "study_type": "Study type",
            "year": "Year",
            "main_finding": "Main finding",
            "stance": "Stance",
            "supporting_or_contradicting": "Supporting / contradicting",
            "limitations": "Limitations",
        }
    )
    st.dataframe(frame, **stretch_kwargs(), hide_index=True)
    st.caption(
        "Stance is detected from the passage's reported direction of effect relative to the "
        "question; it is a transparent lexical signal, not a judgement of scientific truth."
    )
