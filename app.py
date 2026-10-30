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
from utils.logging_config import redact  # noqa: E402

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
    """Cache key component that changes with model/configuration, never the API key."""

    return "|".join(
        [
            "key" if settings.has_llm_key else "nokey",
            settings.llm_provider,
            settings.llm_model,
            str(settings.llm_temperature),
            str(settings.llm_max_output_tokens),
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
# TOP NAVIGATION + MULTI-PAGE INTERFACE
#
# Splice instructions:
# 1. Keep your existing imports, constants, helper functions, run_query(),
#    _build_settings(), _fingerprint(), st.set_page_config(), CSS setup,
#    session-state initialization, and pending-question handling.
# 2. Replace the old UI section beginning at the old `render_sidebar()` call
#    (or the old `# Sidebar` section) through the end of app.py with this block.
# 3. This block expects the functions/classes already defined in your prefix:
#    render_header, render_metrics, render_answer, render_strength,
#    render_citations, render_retrieval_details, render_comparison,
#    render_evidence_assessment, render_notices, render_outcome_banner,
#    backend_status, generation_status, clear_all_caches, redact,
#    run_query, _build_settings, _fingerprint, SOURCE_OPTIONS, EXAMPLES,
#    APP_VERSION, Settings, EvidenceBotResult.
# ---------------------------------------------------------------------------

# Top navigation styling. The radio control remains accessible and functional
# if a Streamlit version changes the DOM structure used by these selectors.
st.markdown(
    """
    <style>
    div[data-testid="stRadio"] > div {
        gap: 0.45rem;
        flex-wrap: wrap;
    }
    div[data-testid="stRadio"] label {
        background: rgba(128, 128, 128, 0.09);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 999px;
        padding: 0.42rem 0.95rem;
        margin: 0;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: rgba(30, 130, 90, 0.16);
        border-color: rgba(30, 130, 90, 0.65);
    }
    div[data-testid="stRadio"] > label {
        display: none;
    }
    .page-intro {
        color: var(--text-color);
        opacity: 0.72;
        margin-top: -0.35rem;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Persistent UI preferences. These values survive Streamlit reruns.
st.session_state.setdefault("active_page", "Overview")
st.session_state.setdefault("selected_sources", list(SOURCE_OPTIONS))
st.session_state.setdefault("top_k", 8)
st.session_state.setdefault("threshold", 0.35)
st.session_state.setdefault("min_passages", 2)
st.session_state.setdefault("max_per_source", 4)

PAGES = ["Overview", "Research", "Sources", "Evidence", "Settings", "History", "About"]

# Keep the existing header/branding, then show the navigation immediately below.
render_header()

selected_page = st.radio(
    "Navigation",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)
st.divider()


def current_config():
    """Build settings from the current persistent UI selections."""
    return _build_settings(
        sources=st.session_state.selected_sources,
        top_k=st.session_state.top_k,
        threshold=st.session_state.threshold,
        min_passages=st.session_state.min_passages,
        max_per_source=st.session_state.max_per_source,
    )


def show_system_status(settings):
    """Display the current LLM and retrieval backend status."""
    llm_status = generation_status(settings)
    vector_status = backend_status()

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Language model")
        if llm_status.get("llm_key_configured"):
            st.success(
                f'{llm_status.get("provider", "LLM")} · '
                f'{llm_status.get("model", "Configured")}'
            )
        else:
            st.warning("API key not configured — generation may be unavailable.")
    with col2:
        st.caption("Retrieval / vector backend")
        backend_name = (
            "FAISS (in-memory)"
            if vector_status.get("faiss_available")
            else "NumPy fallback"
        )
        st.info(backend_name)


def show_result_notices(result):
    """Render notices and the decision banner consistently across pages."""
    if result is None:
        return
    render_notices(result)
    render_outcome_banner(result)


def queue_example(question: str) -> None:
    st.session_state.pending_question = question
    st.session_state.active_page = "Research"


def load_history_entry(item: Dict[str, Any]) -> None:
    st.session_state.result = item.get("result")
    st.session_state.question_input = item.get("question", "")
    st.session_state.active_page = "Research"


def reset_current_session() -> None:
    st.session_state.result = None
    st.session_state.history = []
    st.session_state.question_input = ""
    st.session_state.last_error = None
    st.session_state.run_requested = False
    st.session_state.active_page = "Overview"
    st.session_state.pop("pending_question", None)

with st.sidebar:
    st.markdown("## 🔬 EvidenceBot")
    st.caption(f"Scientific evidence verification · v{APP_VERSION}")
    st.divider()

    st.markdown("### System status")
    llm_status = generation_status(Settings())
    if llm_status.get("llm_key_configured"):
        provider = llm_status.get("provider", "LLM").upper()
        st.success(f"{provider} API key configured")
        st.caption(f"Model: `{llm_status.get('model', 'Unknown')}`")
    else:
        st.warning("Language-model API key not detected")
    vector_status = backend_status()
    st.caption(
        "Vector store: "
        + ("FAISS (in-memory)" if vector_status["faiss_available"] else "NumPy fallback")
    )
    st.caption(f"Embeddings: `{llm_status['embedding_backend']}`")

    st.divider()
    st.markdown("### Quick settings")
    st.multiselect(
        "Scientific sources",
        options=list(SOURCE_OPTIONS),
        key="selected_sources",
        format_func=lambda source: SOURCE_OPTIONS[source],
    )
    st.slider("Results per source", 1, 20, step=1, key="max_per_source")
    st.slider("Top K passages", 3, 30, step=1, key="top_k")
    st.slider("Relevance threshold", 0.0, 1.0, step=0.05, key="threshold")
    st.slider("Minimum evidence passages", 1, 10, step=1, key="min_passages")

    st.divider()
    st.metric("Questions analysed", len(st.session_state.history))

    st.divider()
    st.markdown("### Quick actions")
    if st.button("Clear caches", key="sidebar_clear_caches", **stretch_kwargs()):
        clear_all_caches()
        run_query.clear()
        st.toast("Application caches cleared.")
    st.button(
        "Reset current session",
        key="sidebar_reset_session",
        on_click=reset_current_session,
        **stretch_kwargs(),
    )

if selected_page == "Overview":
    st.title("EvidenceBot")
    st.markdown(
        '<p class="page-intro">A research assistant that checks claims against '
        "retrieved scientific evidence and abstains when support is insufficient.</p>",
        unsafe_allow_html=True,
    )

    snapshot = st.columns(4)
    snapshot[0].metric("Questions analysed", len(st.session_state.history))
    snapshot[1].metric("Sources selected", len(st.session_state.selected_sources))
    snapshot[2].metric("Passages per search", st.session_state.top_k)
    snapshot[3].metric("Evidence threshold", f"{st.session_state.threshold:.2f}")

    if st.session_state.result is not None:
        result = st.session_state.result
        st.subheader("Latest research result")
        if st.session_state.history:
            st.caption(f'Question: {st.session_state.history[0]["question"]}')
        render_metrics(result)
        show_result_notices(result)
        render_answer(result)
        st.button(
            "Open full evidence assessment",
            key="overview_open_evidence",
            on_click=lambda: st.session_state.update(active_page="Evidence"),
        )
    else:
        st.info(
            "No research question has been analysed in this session yet. "
            "Open the Research tab to submit a question."
        )

    st.subheader("Evidence workflow")
    workflow = st.columns(4)
    workflow_steps = (
        ("01", "Retrieve", "Search selected scientific sources."),
        ("02", "Rank", "Surface relevant passages."),
        ("03", "Assess", "Check evidence sufficiency and quality."),
        ("04", "Answer", "Generate a grounded answer or abstain."),
    )
    for column, (number, title, description) in zip(workflow, workflow_steps):
        with column.container(border=True):
            st.markdown(f"**{number}**")
            st.markdown(f"### {title}")
            st.caption(description)

elif selected_page == "Research":
    st.title("Research")
    st.markdown(
        '<p class="page-intro">Ask a focused research question. '
        "EvidenceBot will retrieve sources, assess support, and report limitations.</p>",
        unsafe_allow_html=True,
    )

    with st.form("research_form", clear_on_submit=False):
        question = st.text_area(
            "Research question",
            key="question_input",
            height=115,
            placeholder="e.g. Does mixed-cultivar wheat planting reduce powdery mildew infection?",
        )
        submitted = st.form_submit_button(
            "Analyse evidence",
            type="primary",
            use_container_width=True,
        )

    st.caption("Or start with an example question")
    example_columns = st.columns(2)
    for index, (example_question, tag) in enumerate(EXAMPLES):
        with example_columns[index % 2]:
            st.button(
                f"{tag} · {example_question}",
                key=f"example_{index}",
                use_container_width=True,
                on_click=queue_example,
                args=(example_question,),
            )

    if submitted:
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            st.warning("Enter a research question before starting the analysis.")
        elif not st.session_state.selected_sources:
            st.warning("Select at least one scientific source in Settings.")
        else:
            settings = current_config()
            st.session_state.last_error = None
            with st.spinner("Retrieving and assessing scientific evidence..."):
                try:
                    result = run_query(
                        cleaned_question,
                        tuple(st.session_state.selected_sources),
                        st.session_state.top_k,
                        st.session_state.threshold,
                        st.session_state.min_passages,
                        st.session_state.max_per_source,
                        _fingerprint(settings),
                    )
                    st.session_state.result = result
                    st.session_state.history.insert(
                        0,
                        {
                            "question": cleaned_question,
                            "result": result,
                            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        },
                    )
                    st.session_state.history = st.session_state.history[:30]
                except Exception as exc:  # noqa: BLE001 - surface errors safely
                    st.session_state.last_error = redact(f"{type(exc).__name__}: {exc}")
                    st.session_state.result = None

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.result is not None:
        st.divider()
        st.subheader("Analysis result")
        result = st.session_state.result
        render_metrics(result)
        show_result_notices(result)
        render_answer(result)
        render_strength(result.strength)
        render_citations(result)
        st.button(
            "View detailed evidence assessment",
            key="research_open_evidence",
            on_click=lambda: st.session_state.update(active_page="Evidence"),
        )


# ---------------------------------------------------------------------------
# SOURCES
# ---------------------------------------------------------------------------
elif selected_page == "Sources":
    st.title("Sources")
    st.markdown(
        '<p class="page-intro">Inspect the references and retrieval information '
        "for the latest analysis.</p>",
        unsafe_allow_html=True,
    )

    result = st.session_state.result
    if result is None:
        st.info("Run a research question first. Its references will appear here.")
    else:
        render_citations(result)
        st.divider()
        st.subheader("Retrieval details")
        render_retrieval_details(result)


# ---------------------------------------------------------------------------
# EVIDENCE
# ---------------------------------------------------------------------------
elif selected_page == "Evidence":
    st.title("Evidence assessment")
    st.markdown(
        '<p class="page-intro">Review the evidence-strength decision, '
        "supporting passages, comparison, and reasons for any abstention.</p>",
        unsafe_allow_html=True,
    )

    result = st.session_state.result
    if result is None:
        st.info("There is no evidence assessment yet. Run a research question first.")
    else:
        show_result_notices(result)
        render_strength(result.strength)
        render_evidence_assessment(result)
        render_comparison(result)
        render_retrieval_details(result)


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
elif selected_page == "Settings":
    st.title("Settings")
    st.markdown(
        '<p class="page-intro">Configure retrieval and evidence-ranking behaviour. '
        "Changes apply to the next analysis.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Scientific sources")
    st.info("Adjust scientific sources and retrieval parameters from the sidebar.")
    st.write(
        ", ".join(
            SOURCE_OPTIONS[source]
            for source in st.session_state.selected_sources
        )
        or "No sources selected"
    )
    st.caption(
        f"{st.session_state.top_k} passages · threshold "
        f"{st.session_state.threshold:.2f} · "
        f"{st.session_state.min_passages} minimum evidence passages · "
        f"{st.session_state.max_per_source} results per source"
    )

    st.divider()
    st.subheader("Model and backend status")
    settings = current_config()
    show_system_status(settings)

    st.subheader("API key configuration")
    st.markdown(
        """
        Configure your provider key in the environment used to run Streamlit
        (for example, Codespaces Secrets or your local `.env` file). Do not paste
        API keys into this page or commit them to GitHub.

        For a Groq setup, the environment variable is commonly
        `GROQ_API_KEY`; use the exact variable name expected by your project's
        `config.py`. After changing a secret, restart or rerun the app as required
        by your environment.
        """
    )

    st.divider()
    st.subheader("Maintenance")
    if st.button("Clear application caches", type="secondary"):
        try:
            clear_all_caches()
            st.success("Application caches cleared.")
        except Exception as exc:
            st.error(f"Could not clear caches: {redact(str(exc))}")


# ---------------------------------------------------------------------------
# HISTORY
# ---------------------------------------------------------------------------
elif selected_page == "History":
    st.title("Research history")
    st.markdown(
        '<p class="page-intro">Questions analysed during this Streamlit session.</p>',
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.info("No questions have been analysed in this session.")
    else:
        st.caption(f"{len(history)} question(s) in this session")
        for index, item in enumerate(history):
            question_text = item.get("question", "Untitled question")
            timestamp = item.get("timestamp", "")
            with st.expander(f"{timestamp} — {question_text}", expanded=(index == 0)):
                st.button(
                    "Load this result",
                    key=f"history_load_{index}",
                    on_click=load_history_entry,
                    args=(item,),
                )
        if st.button("Clear session history", type="secondary"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif selected_page == "About":
    st.title("About EvidenceBot")
    st.markdown(
        """
        EvidenceBot is designed to support research exploration by retrieving
        scientific literature, ranking relevant material, assessing whether the
        retrieved evidence is sufficient, and generating answers grounded in
        the available sources.

        **Typical pipeline**

        1. **Retrieval:** query configured literature and reference services.
        2. **Processing:** normalise, deduplicate, and prepare retrieved records.
        3. **Ranking:** identify relevant documents and passages.
        4. **Evidence gate:** determine whether the available material meets the
           configured support requirements.
        5. **Generation and validation:** produce a grounded response and check
           its citations, or abstain when support is insufficient.

        **Important limitations**

        - Retrieved literature may be incomplete or unavailable.
        - A relevance score is not proof that a claim is true.
        - EvidenceBot is a research aid, not a substitute for reading and
          critically evaluating the original papers.
        - Session history is stored in Streamlit session state and is not a
          permanent database.
        """
    )
    st.caption(f"EvidenceBot version {APP_VERSION}")


# Consistent footer across all pages.
st.divider()
st.caption(
    "EvidenceBot · Grounded research assistance · Always verify important claims "
    "against the original scientific publications."
)
