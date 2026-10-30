"""EvidenceBot — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure the repository root is importable when launched from elsewhere.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, ROOT)

from config import Settings  # noqa: E402
from generation.llm import generation_status  # noqa: E402
from pipeline import EvidenceBot, EvidenceBotResult  # noqa: E402
from ranking.vector_store import backend_status  # noqa: E402
from ui.components import (  # noqa: E402
    stretch_kwargs,
    render_answer,
    render_citations,
    render_comparison,
    render_evidence_assessment,
    render_header,
    render_metrics,
    render_notices,
    render_outcome_banner,
    render_retrieval_details,
    render_strength,
    section,
)
from ui.styles import CSS  # noqa: E402
from utils.cache import clear_all_caches  # noqa: E402

APP_VERSION = "1.0.0"

SOURCE_OPTIONS = {
    "pubmed": "PubMed / NCBI",
    "europe_pmc": "Europe PMC",
    "arxiv": "arXiv",
    "crossref": "Crossref",
    "wikipedia": "Wikipedia (background only)",
}

EXAMPLES: List[Tuple[str, str]] = [
    ("What role does IL-6 play in rheumatoid arthritis?", "Blood: immunology"),
    ("Does vitamin D supplementation reduce cancer risk?", "Nutrition: oncology"),
    (
        "Do omega-3 fatty acids reduce cardiovascular risk?",
        "Cardiology: conflicting trials",
    ),
    (
        "Does drinking a glass of orange juice every morning guarantee that a person will never "
        "develop cancer?",
        "Unanswerable: absolute claim",
    ),
    (
        "What role does IL-6 play in rheumatoid arthritis and what is the long-term effect of "
        "tocilizumab on all-cause mortality?",
        "Partially answerable",
    ),
    ("Is the gut microbiome a cause of Parkinson's disease?", "Neuroscience"),
]


# ---------------------------------------------------------------------------
# Cached pipeline entry point
# ---------------------------------------------------------------------------


def _build_settings(
    sources: Tuple[str, ...],
    top_k: int,
    threshold: float,
    min_passages: int,
    max_per_source: int,
) -> Settings:
    base = Settings()
    return base.with_overrides(
        enabled_sources=tuple(sources),
        top_k=int(top_k),
        relevance_threshold=float(threshold),
        minimum_evidence_passages=int(min_passages),
        max_results_per_source=int(max_per_source),
    )


@st.cache_data(ttl=1800, max_entries=48, show_spinner=False)
def run_query(
    question: str,
    sources: Tuple[str, ...],
    top_k: int,
    threshold: float,
    min_passages: int,
    max_per_source: int,
    config_fingerprint: str,
) -> EvidenceBotResult:
    """Run the full EvidenceBot pipeline (cached for identical questions/config)."""

    settings = _build_settings(sources, top_k, threshold, min_passages, max_per_source)
    engine = EvidenceBot(settings)
    return engine.ask(
        question,
        sources=list(sources),
        top_k=top_k,
        threshold=threshold,
        minimum_passages=min_passages,
    )


def _fingerprint(settings: Settings) -> str:
    """Cache key component that changes with model/key availability, never the key."""

    return "|".join(
        [
            "key" if settings.has_openai_key else "nokey",
            settings.openai_model,
            settings.openai_embedding_model,
            settings.resolved_embedding_backend(),
            str(settings.chunk_size_words),
            str(settings.claim_support_threshold),
        ]
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="EvidenceBot — scientific evidence verification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "run_requested" not in st.session_state:
    st.session_state.run_requested = False

# A pending question is applied *before* the text input widget is instantiated,
# which is the only point at which a widget's session-state key may be written.
_pending_question = st.session_state.pop("pending_question", None)
if _pending_question is not None:
    st.session_state.question_input = _pending_question


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> Dict[str, Any]:
    settings = Settings()
    with st.sidebar:
        st.markdown("## 🔬 EvidenceBot")
        st.caption("Scientific evidence verification · v" + APP_VERSION)

        st.markdown("### Retrieval sources")
        selected_sources = st.multiselect(
            "Databases queried concurrently",
            options=list(SOURCE_OPTIONS.keys()),
            default=list(SOURCE_OPTIONS.keys()),
            format_func=lambda key: SOURCE_OPTIONS[key],
            help="Wikipedia is background material only and is never cited as scientific proof.",
        )
        max_per_source = st.slider(
            "Results per source", min_value=3, max_value=20, value=8, step=1,
            help="How many records each API may return.",
        )

        st.markdown("### Ranking and evidence gate")
        top_k = st.slider("TOP_K — passages kept", 3, 20, 10, 1)
        threshold = st.slider(
            "RELEVANCE_THRESHOLD", 0.05, 0.90, 0.30, 0.05,
            help="Hybrid relevance a passage must reach to count as on-topic.",
        )
        min_passages = st.slider(
            "MINIMUM_EVIDENCE_PASSAGES", 1, 5, 2, 1,
            help="On-topic passages required before answering. Higher = stricter abstention.",
        )
        st.caption(
            "These parameters map directly to `config.py` / environment variables. "
            "A stricter threshold means more abstentions — the intended trade-off."
        )

        st.markdown("### Model status")
        status = generation_status(settings)
        if status["openai_key_configured"]:
            st.success(f"OpenAI key detected · `{status['model']}`")
        else:
            st.warning(
                "No OPENAI_API_KEY detected. Retrieval, ranking and the evidence gate still "
                "work; answers are replaced by a verbatim extractive digest."
            )
        st.caption(
            f"Embeddings: `{status['embedding_backend']}`"
            + (
                f" ({status['embedding_model']})"
                if status["embedding_backend"] == "openai"
                else " (offline hashed features)"
            )
        )
        vector_backend = backend_status()
        st.caption(
            "Vector store: "
            + ("FAISS (in-memory)" if vector_backend["faiss_available"] else "NumPy fallback")
        )

        st.markdown("### Maintenance")
        if st.button("Clear cached API/LLM responses", **stretch_kwargs()):
            clear_all_caches()
            run_query.clear()
            st.toast("Caches cleared.")

        with st.expander("Where do I put my API key?", expanded=False):
            st.markdown(
                "Create a file named `.env` in the project root:\n\n"
                "```\nOPENAI_API_KEY=sk-...\n```\n\n"
                "or use `.streamlit/secrets.toml`:\n\n"
                "```\nOPENAI_API_KEY = \"sk-...\"\n```\n\n"
                "The key is read from the environment only. It is never written to logs, "
                "displayed in the interface, or committed (`.env` and "
                "`.streamlit/secrets.toml` are git-ignored)."
            )

        with st.expander("What EvidenceBot will not do", expanded=False):
            st.markdown(
                "- It will not answer without retrieved, on-topic evidence.\n"
                "- It will not invent citations, DOIs or URLs.\n"
                "- It will not call the LLM when the evidence gate abstains.\n"
                "- It will not treat Wikipedia or preprints as equal to peer-reviewed research.\n"
                "- It will not interpret missing evidence as proof that a claim is false."
            )

        st.markdown("---")
        st.caption("From real literature → to trusted answers.")

    return {
        "sources": tuple(selected_sources),
        "top_k": top_k,
        "threshold": threshold,
        "min_passages": min_passages,
        "max_per_source": max_per_source,
    }


config = render_sidebar()


# ---------------------------------------------------------------------------
# Header + search
# ---------------------------------------------------------------------------

render_header()
section("1", "Ask a scientific question")

with st.form("search_form", clear_on_submit=False):
    question = st.text_input(
        "Research question",
        key="question_input",
        placeholder="e.g. What role does IL-6 play in rheumatoid arthritis?",
        label_visibility="collapsed",
    )
    col_search, col_clear, _ = st.columns([1.2, 0.8, 3])
    submitted = col_search.form_submit_button("🔍 Search evidence", **stretch_kwargs(), type="primary")
    cleared = col_clear.form_submit_button("✕ Clear", **stretch_kwargs())

st.markdown(
    '<div class="eb-small" style="margin-top:-6px">Try an example:</div>',
    unsafe_allow_html=True,
)
example_columns = st.columns(3)
for index, (example_question, tag) in enumerate(EXAMPLES):
    column = example_columns[index % 3]
    short = example_question if len(example_question) <= 72 else example_question[:69] + "…"
    if column.button(
        f"{tag} · {short}",
        key=f"example_{index}",
        **stretch_kwargs(),
        help=example_question,
    ):
        st.session_state.pending_question = example_question
        st.session_state.run_requested = True
        st.rerun()

if cleared:
    st.session_state.result = None
    st.session_state.last_error = None
    # Applied before the widget is re-instantiated on the next run.
    st.session_state.pending_question = ""
    st.rerun()

should_run = submitted or st.session_state.pop("run_requested", False)
if should_run:
    query = (question or st.session_state.get("question_input") or "").strip()
    if not query:
        st.warning("Enter a research question first.")
    elif not config["sources"]:
        st.error("Select at least one retrieval source in the sidebar.")
    else:
        settings = _build_settings(
            config["sources"], config["top_k"], config["threshold"],
            config["min_passages"], config["max_per_source"],
        )
        with st.spinner(
            "Searching PubMed, Europe PMC, arXiv, Crossref and Wikipedia · ranking passages · "
            "evaluating evidence sufficiency…"
        ):
            try:
                result = run_query(
                    query,
                    config["sources"],
                    config["top_k"],
                    config["threshold"],
                    config["min_passages"],
                    config["max_per_source"],
                    _fingerprint(settings),
                )
                st.session_state.result = result
                st.session_state.last_error = None
                st.session_state.history.insert(
                    0,
                    {
                        "question": query,
                        "run_id": result.run_id,
                        "time": dt.datetime.now().strftime("%H:%M:%S"),
                        "decision": result.decision,
                        "strength": result.strength.grade,
                        "citations": result.citations_used,
                        "documents": result.documents_retrieved,
                    },
                )
                # Keep one entry per question, most recent first.
                seen_questions = set()
                deduped = []
                for entry in st.session_state.history:
                    if entry["question"] in seen_questions:
                        continue
                    seen_questions.add(entry["question"])
                    deduped.append(entry)
                st.session_state.history = deduped[:12]
            except Exception as exc:  # noqa: BLE001 - surface errors, never crash
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"
                st.session_state.result = None
                if os.getenv("EVIDENCEBOT_DEBUG"):
                    st.exception(exc)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

result: Optional[EvidenceBotResult] = st.session_state.result

if st.session_state.last_error:
    st.error(
        "The search could not be completed: "
        + st.session_state.last_error
        + "\n\nThis is usually a network or API-key problem. Check the source status panel "
        "and your connection, then try again."
    )

if result is None:
    section("2", "How EvidenceBot decides")
    st.markdown(
        """
<div class="eb-status">
  <div class="eb-status__body">
    <b>1. Retrieve.</b> All selected databases are queried concurrently; failures in one source
    never break the search.<br>
    <b>2. Standardise &amp; de-duplicate.</b> Records become a unified document table, then
    sentence-aware chunks that keep their parent metadata.<br>
    <b>3. Rank.</b> Dense embeddings in FAISS + BM25 keyword search + concept coverage produce a
    hybrid relevance score per passage.<br>
    <b>4. Evaluate.</b> A deterministic evidence gate checks relevance, sufficiency, source
    quality and consistency. It can refuse to answer.<br>
    <b>5. Generate.</b> Only if the gate approves: the LLM may use nothing except the numbered
    evidence passages.<br>
    <b>6. Verify.</b> Every citation is mapped back to a retrieved source; unsupported or invented
    citations are removed and the answer is flagged.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eb-notice eb-notice--info"><b>No evidence, no answer.</b> '
        "If the retrieved passages cannot support the question, EvidenceBot returns exactly: "
        "“Insufficient evidence. No reliable sources found in the retrieved passages.”</div>",
        unsafe_allow_html=True,
    )
else:
    section("2", "Evidence decision")
    render_outcome_banner(result)
    render_notices(result)

    section("3", "Answer")
    render_answer(result)

    if not result.is_abstention:
        section("4", "Evidence strength")
        render_strength(result.strength)

    section("5", "Citations and sources")
    render_citations(result)

    section("6", "Retrieval analytics")
    render_metrics(result)
    st.caption(
        "Documents retrieved are all unique publications returned by the databases. "
        "Passages used as evidence are the only ones the answer may cite."
    )
    render_retrieval_details(result)

    section("7", "Evidence comparison across publications")
    render_comparison(result)
    render_evidence_assessment(result)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

section("8", "Search history (this session)")
if not st.session_state.history:
    st.caption("No searches yet in this session.")
else:
    for history_index, item in enumerate(st.session_state.history):
        icon = {"ANSWERABLE": "✅", "PARTIALLY_ANSWERABLE": "⚠️", "UNANSWERABLE": "⛔"}.get(
            item["decision"], "•"
        )
        columns = st.columns([0.6, 5.2, 1.4, 1.0, 1.2])
        columns[0].markdown(f"**{icon}**")
        columns[1].markdown(item["question"])
        columns[2].caption(f"{item['strength']} · {item['citations']} citations")
        columns[3].caption(item["time"])
        if columns[4].button("Re-run", key=f"rerun_{history_index}_{item['time']}"):
            st.session_state.pending_question = item["question"]
            st.session_state.run_requested = True
            st.rerun()
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()


section("9", "About this build")
with st.expander("Architecture, guarantees and known limitations", expanded=False):
    st.markdown(
        """
**Pipeline.** `retrieval/` (PubMed E-utilities via Biopython, Europe PMC, arXiv, Crossref,
Wikipedia) → `processing/` (cleaning, de-duplication, chunking) → `ranking/` (OpenAI or offline
embeddings, FAISS, BM25, hybrid reranker) → `evidence/` (evidence gate, abstention, citation
validator, strength grading) → `generation/` (OpenAI structured JSON) → `ui/`.

**Guarantees.**
- The evidence gate runs before any generation call and cannot be overridden by the LLM.
- Abstention returns the exact message: *“Insufficient evidence. No reliable sources found in
  the retrieved passages.”*
- Citation numbers are re-mapped to retrieved passages only; unknown ids, unsupported claims,
  unknown DOIs and URLs are rejected and removed.
- Evidence strength is computed by explicit rules, never by the model. It is not a probability.
- Wikipedia and preprints are labelled and down-weighted; background material never satisfies
  the evidence requirement.

**Known limitations.**
- Retrieval only sees abstracts unless open-access full text is available (Europe PMC).
- Consistency and claim support use transparent lexical heuristics, so paraphrases can be
  flagged conservatively — the system prefers withholding a claim to overstating one.
- Crossref often has no abstract; such records are marked *metadata only*.
- A single weak source cannot produce an answer (MINIMUM_EVIDENCE_PASSAGES), which can look
  strict on narrow questions.
- Without `OPENAI_API_KEY` the app still retrieves, ranks and gated-decides, but shows a verbatim
  extractive digest instead of a synthesised answer.
- EvidenceBot never states that a claim is *false* because evidence is missing.
        """
    )

st.markdown(
    '<div class="eb-footer">From real literature &nbsp;→&nbsp; to <b>trusted answers</b> · '
    "Cite. Verify. Be honest.</div>",
    unsafe_allow_html=True,
)
