# EvidenceBot

**Multi-source scientific evidence retrieval and verification system.**

> PubMed-grounded answers that cite their sources and abstain when evidence is missing.
> **No evidence → no answer.**

EvidenceBot is a Streamlit application that answers biomedical and scientific questions from
real scientific literature. Unlike a general chatbot it retrieves from several databases at
once, ranks the retrieved passages, decides whether the retrieved evidence is actually
sufficient, and only then writes an answer — with every claim tied to a retrieved source. When
the evidence does not support the question it abstains and returns exactly:

> **Insufficient evidence. No reliable sources found in the retrieved passages.**

---

## 1. Quick start

```bash
# 1. Clone and enter the project
git clone <your-repo-url> EvidenceBot
cd EvidenceBot

# 2. Create a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
#    If `faiss-cpu` has no wheel for your platform, install everything else and
#    skip FAISS — the app automatically falls back to an exact NumPy vector store:
#    pip install streamlit pandas numpy requests biopython rank-bm25 openai python-dotenv pytest

# 4. Add your OpenAI API key (see section 3)
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY=sk-...

# 5. Launch the dashboard
streamlit run app.py
```

The app opens at <http://localhost:8501>.

**Minimum supported command for a headless/proxied environment:**

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### No API key? It still works, without inventing anything

Without `OPENAI_API_KEY` the whole retrieval → ranking → evidence-gate pipeline still runs, and
the answer area shows a **verbatim extractive digest**: quoted sentences from the retrieved
passages with their citation numbers. Nothing is paraphrased or generated, so nothing can be
hallucinated. Semantic embeddings automatically fall back to an offline hashed-feature embedder
(`EMBEDDING_BACKEND=auto`).

---

## 2. What it does

| Stage | Module | What happens |
|---|---|---|
| 1. Retrieve | `retrieval/` | PubMed/NCBI E-utilities, Europe PMC, arXiv, Crossref and Wikipedia are queried **concurrently** with timeouts, retries, per-host rate limiting and response caching. One failing API never breaks the search. |
| 2. Standardise | `processing/` | Records are cleaned into a unified schema, de-duplicated across sources (DOI/PMID/arXiv id/title similarity, peer-reviewed record wins) and stored in a Pandas DataFrame. |
| 3. Chunk | `processing/chunking.py` | Sentence-aware chunks (size/overlap configurable); every chunk keeps its parent document id and metadata. Unrelated publications are never merged. |
| 4. Rank | `ranking/` | Hybrid retrieval: OpenAI embeddings + FAISS (in-memory, NumPy fallback) **and** BM25 keyword search **and** question-concept coverage → one relevance score per passage, with a per-publication diversity cap. |
| 5. Evaluate | `evidence/evidence_evaluator.py` | Deterministic gate over four dimensions: retrieval relevance, evidence sufficiency, source quality, evidence consistency. Outputs `ANSWERABLE`, `PARTIALLY_ANSWERABLE` or `UNANSWERABLE`. |
| 6. Generate | `generation/` | Only if the gate approves. OpenAI produces structured JSON (answer, supporting claims + source ids, evidence assessment, limitations, unanswered questions, conflicts). |
| 7. Verify | `evidence/citation_validator.py` | Every `[n]` is mapped back to a retrieved passage; unknown ids, invented URLs/DOIs, unsupported claims and causal overstatements are detected; the answer is repaired once, then flagged and sanitised if it still fails. |
| 8. Grade | `evidence/evidence_strength.py` | Transparent rule-based evidence strength: Strong / Moderate / Limited / Conflicting / Insufficient. Never produced by the LLM, never a probability. |

### The abstention rules (they cannot be overridden)

The generation call is never made when the gate abstains. The gate abstains when:

- no passage clears `RELEVANCE_THRESHOLD`;
- fewer than `MINIMUM_EVIDENCE_PASSAGES` on-topic passages remain;
- the usable evidence is too short (`MIN_EVIDENCE_WORDS`);
- the question asks for a **universal guarantee** ("never", "guarantee", "100%") that no finite
  study can establish, unless a retrieved study directly reports a null result on the same
  exposure/outcome;
- the passages are topically related but do not address the question's substance
  (question-concept coverage fails).

`PARTIALLY_ANSWERABLE` is returned when at least one component of a compound question is
supported and others are not: the supported part is answered, and the missing part is listed
explicitly under "unanswered parts of the question".

The system never treats *missing* evidence as proof that a claim is false.

---

## 3. Where to put your OpenAI API key

EvidenceBot reads the key **only** from the environment or Streamlit secrets. It is never
hard-coded, never logged, never displayed.

**Option A — `.env` (recommended for local use)**

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Option B — Streamlit secrets**

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-key-here"
```

**Option C — shell environment**

```bash
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

Both `.env` and `.streamlit/secrets.toml` are git-ignored. The sidebar shows whether a key was
detected and which models are in use; the key value itself is never shown.

---

## 4. Configuration

Every knob lives in `config.py` and can be set as an environment variable (see `.env.example`).
The most useful ones are also exposed as sidebar sliders and passed straight through to the
pipeline.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Answer-generation model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_BACKEND` | `auto` | `auto` / `openai` / `local` (offline hashed features) |
| `MAX_RESULTS_PER_SOURCE` | `8` | Records requested per database |
| `TOP_K` | `10` | Passages kept after reranking |
| `RELEVANCE_THRESHOLD` | `0.30` | Minimum hybrid relevance for a passage to count |
| `MINIMUM_EVIDENCE_PASSAGES` | `2` | On-topic passages required to answer |
| `MIN_QUESTION_COVERAGE` | `0.55` | Question concepts that must appear in the evidence |
| `MIN_EVIDENCE_WORDS` | `45` | Minimum volume of usable evidence text |
| `CLAIM_SUPPORT_THRESHOLD` | `0.45` | Share of a claim's terms that its cited passage must contain |
| `MAX_GENERATION_ATTEMPTS` | `2` | Generation + one citation-repair attempt |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `170` / `40` | Chunking parameters |
| `REQUEST_TIMEOUT_SECONDS` / `REQUEST_RETRIES` | `12` / `2` | HTTP behaviour |
| `CACHE_TTL_SECONDS` | `3600` | API/LLM response cache lifetime (`.cache/evidencebot/`) |
| `NCBI_EMAIL`, `NCBI_API_KEY`, `CROSSREF_MAILTO` | — | Polite API identification; NCBI key raises the rate limit to 10 req/s |
| `ENABLE_OFFLINE_SAMPLE_FALLBACK` | `true` | Load the labelled illustrative corpus **only** if every live source is unreachable |
| `ABSOLUTE_CLAIM_POLICY` | `abstain` | Behaviour for universal-claim questions |

Set `MINIMUM_EVIDENCE_PASSAGES=1` for a less strict tool; set it to 3–4 for a stricter one.
Raising `RELEVANCE_THRESHOLD` increases abstentions — that is the intended trade-off.

---

## 5. Project structure

```
EvidenceBot/
├── app.py                      # Streamlit dashboard
├── config.py                   # settings, env/secrets resolution, abstention message
├── pipeline.py                 # orchestration: retrieve → gate → generate → verify
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── .streamlit/config.toml      # theme + server settings
├── data/offline_sample_corpus.json   # clearly-labelled offline demo corpus
├── retrieval/
│   ├── base.py                 # HTTP client (timeouts, retries, rate limits, cache)
│   ├── pubmed.py               # NCBI E-utilities + Biopython MEDLINE parsing
│   ├── europe_pmc.py           # Europe PMC REST + open-access full text
│   ├── arxiv.py                # arXiv API, in-scope categories only
│   ├── crossref.py             # Crossref metadata
│   ├── wikipedia.py            # background material only
│   └── retrieval_manager.py    # concurrent fan-out, failure recording, fallbacks
├── processing/
│   ├── models.py               # Document / Passage, source taxonomy and weights
│   ├── document_processor.py   # cleaning, DataFrame, dedup, chunking
│   ├── chunking.py             # sentence-aware overlapping chunks
│   └── deduplication.py        # cross-source de-duplication and merging
├── ranking/
│   ├── embeddings.py           # OpenAI embeddings + offline hashed embedder, caching
│   ├── vector_store.py         # FAISS in-memory store with NumPy fallback
│   ├── bm25.py                 # BM25 index
│   └── reranker.py             # hybrid scoring and reranking
├── evidence/
│   ├── evidence_evaluator.py   # four-dimension gate (separate, testable module)
│   ├── abstention.py           # binding decisions and the exact message
│   ├── citation_validator.py   # citation/claim/URL/DOI verification
│   ├── evidence_strength.py    # rule-based strength grading
│   └── stance.py               # direction-of-effect detection and consistency
├── generation/
│   ├── prompts.py              # strict grounded-generation prompts
│   ├── llm.py                  # OpenAI client wrapper (injectable for tests)
│   └── extractive.py           # verbatim digest when no LLM is available
├── utils/
│   ├── logging_config.py       # logging with API-key redaction
│   ├── helpers.py              # text utilities, term extraction, question parsing
│   └── cache.py                # TTL + disk cache
├── ui/
│   ├── styles.py               # poster-inspired visual identity
│   └── components.py           # header, banners, citations, analytics, comparison
└── tests/
    ├── conftest.py             # offline settings, fake HTTP, fake LLM, builders
    ├── fixtures/               # recorded API payloads (PubMed, EPMC, arXiv, Crossref, Wikipedia)
    ├── test_retrieval.py
    ├── test_abstention.py
    ├── test_citations.py
    ├── test_generation.py
    ├── test_evidence_evaluator.py
    └── test_pipeline.py        # end-to-end scenarios 1-7
```

---

## 6. How the interface is organised

1. **Ask a scientific question** — free text, example questions, clear button, loading spinner.
2. **Evidence decision** — green (answerable), orange (partial), purple (abstained) card, with the
   exact insufficient-evidence message when abstaining.
3. **Answer** — grounded answer with inline `[n]` citations, or the verbatim extractive digest.
   Supporting claims, limitations and unanswered parts are collapsible panels.
4. **Evidence strength** — grade plus a full breakdown of the rule-based components.
5. **Citations and sources** — one card per citation: number, title, source, authors, year,
   journal, DOI, PMID, clickable URL and the exact supporting passage.
6. **Retrieval analytics** — documents retrieved vs. passages used as evidence, relevant
   passages, sources accessed, hybrid similarity scores, citations used, retrieval/ranking time.
7. **Evidence comparison** — table of publication, study type, main finding, stance
   (supports / contradicts / no effect) and limitations, to expose scientific disagreement.
8. **Search history** — session-only, with re-run and clear.
9. **About** — architecture, guarantees and known limitations.

Higher relevance thresholds and higher `MINIMUM_EVIDENCE_PASSAGES` in the sidebar produce more
abstentions by design.

---

## 7. Testing

```bash
pytest -q                     # full suite
pytest tests/test_abstention.py -v
```

No test touches the network or a paid API: retrieval is served from `tests/fixtures/`, and the
OpenAI client is replaced by an injectable fake. The suite covers the seven required scenarios:

| Scenario | Test |
|---|---|
| 1. Answerable question (IL-6 in RA) | `test_pipeline.py::test_answerable_question_end_to_end` |
| 2. Unanswerable absolute claim (orange juice) | `test_abstention.py::test_pipeline_abstains_on_absolute_claim_with_real_passages`, `test_pipeline.py::test_unanswerable_question_end_to_end` |
| 3. Partially answerable question | `test_abstention.py::test_partially_answerable_question_generates_with_gaps` |
| 4. Invalid citation id | `test_citations.py::test_invalid_citation_is_rejected_and_removed` |
| 5. Retrieval API failure | `test_retrieval.py::test_manager_survives_pubmed_outage`, `test_pipeline.py::test_source_failure_does_not_break_pipeline` |
| 6. Conflicting findings | `test_pipeline.py::test_conflicting_findings_are_surfaced_to_the_model` |
| 7. Hallucination prevention | `test_generation.py::test_pipeline_flags_unverifiable_answer`, `test_citations.py::test_unsupported_claim_is_detected` |

Additional tests cover de-duplication, preprint/background handling, chunking, source parsing,
prompt construction, evidence-strength rules and abstention-without-LLM-call.

---

## 8. Suggested questions

**Answerable.** *What role does IL-6 play in rheumatoid arthritis?* — mechanistic reviews, RCTs
and a meta-analysis are retrieved; the answer cites each claim and grades the evidence.

**Partially answerable.** *What role does IL-6 play in rheumatoid arthritis and what is the
long-term effect of tocilizumab on all-cause mortality?* — the IL-6 component is answered; the
mortality component is reported as missing evidence.

**Unanswerable.** *Does drinking a glass of orange juice every morning guarantee that a person
will never develop cancer?* — the gate refuses the absolute claim and returns the exact
insufficient-evidence message without calling the LLM.

**Conflicting.** *Do omega-3 fatty acids reduce cardiovascular risk?* — trials and
meta-analyses disagree; the answer must present the disagreement, and the strength grade is
"Conflicting evidence".

---

## 9. Verified behaviour in this repository

- `pytest -q` → all tests pass (no network, no API key required).
- The dashboard boots with or without an API key; the sidebar reports model/embedding/vector
  backend status and where to put the key.
- Source failures (simulated PubMed outage) are surfaced as a warning while the remaining
  sources continue.
- Citation validation removes invalid ids and unsupported claims, and the response is labelled
  as not fully verified instead of being presented as evidence-backed.

---

## 10. Known limitations

1. **Abstracts, not full text.** Retrieval mostly sees abstracts; Europe PMC open-access full
   text is fetched for a couple of records per search (`MAX_FULLTEXT_FETCHES`).
2. **Lexical heuristics for claim support and consistency.** Claim-to-passage support, direction
   of effect and conflict detection are transparent lexical rules. They are deliberately strict:
   genuine paraphrases can be flagged as "not cross-checked", and the system prefers withholding
   a claim to overstating one. Embedding-based entailment would reduce false positives but adds
   cost and non-determinism.
3. **Coverage limits of the sources.** Crossref records frequently have no abstract (marked
   *metadata only*), arXiv covers preprints rather than peer-reviewed work, and PubMed's
   relevance ranking is not domain-optimised for every question.
4. **Strict default.** With `MINIMUM_EVIDENCE_PASSAGES=2`, a question with one excellent but
   isolated source is refused. Lower it to 1 for a more permissive tool.
5. **No medical advice.** EvidenceBot summarises literature; it is not a diagnostic or treatment
   recommendation system and must not be used for clinical decisions.
6. **Offline sample corpus.** If *every* live source is unreachable (e.g. an air-gapped sandbox)
   and `ENABLE_OFFLINE_SAMPLE_FALLBACK=true`, the app loads
   `data/offline_sample_corpus.json` — hand-written illustrative records with no DOI/PMID/URL.
   They are flagged `is_sample`, banner-labelled in the UI and marked in citation cards. They are
   **not real literature**; set the flag to `false` to disable the fallback entirely.
7. **Retrieval scores are not truth.** A high hybrid relevance score means a passage matches the
   question, not that the science is correct, sufficient or uncontested. Sufficiency is decided
   by the evidence gate; evidence strength is a rule-based index, not a probability.
8. **LLM dependence for prose.** With a key, the answer wording comes from the model, but it may
   only use the numbered passages and is post-validated. Without a key there is no synthesised
   answer at all, only verbatim quotes.

---

## 11. License and attribution

Built for the AI4BIOMED / BIOAI Hackathon (Vignan's Foundation for Science, Technology &
Research) — team *Lifetech Innovators*. Data sources: NCBI PubMed, Europe PMC, arXiv, Crossref,
Wikipedia. Please respect each provider's terms of use and rate limits (set `NCBI_EMAIL`,
`NCBI_API_KEY` and `CROSSREF_MAILTO` to identify your deployment).

**From real literature → to trusted answers. Cite. Verify. Be honest.**
