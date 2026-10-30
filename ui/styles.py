"""Visual identity for the EvidenceBot dashboard.

Navy + white base, cyan/light-blue accents, red editorial accents, and semantic
colours for outcomes (green = supported, orange = partial, purple/red = abstain).
The CSS is scoped to custom classes so it cannot break Streamlit internals.
"""

from __future__ import annotations

CSS = """
<style>
:root {
  --eb-navy: #0a1a33;
  --eb-navy-2: #0f2748;
  --eb-navy-soft: #12294f;
  --eb-blue: #1d5fd6;
  --eb-cyan: #22c9e6;
  --eb-cyan-soft: #d9f4fb;
  --eb-red: #d7263d;
  --eb-green: #12805c;
  --eb-green-soft: #e6f6ef;
  --eb-orange: #b45309;
  --eb-orange-soft: #fdf3e3;
  --eb-purple: #6b21a8;
  --eb-purple-soft: #f4ebfe;
  --eb-ink: #0f1b2d;
  --eb-muted: #5a6b83;
  --eb-line: #dbe4ef;
}

/* ---------- page frame ---------- */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(34, 201, 230, 0.10), transparent 60%),
    radial-gradient(900px 400px at 5% 0%, rgba(29, 95, 214, 0.08), transparent 55%),
    #f7fafd;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1280px; }

/* ---------- hero ---------- */
.eb-hero {
  position: relative;
  overflow: hidden;
  border-radius: 20px;
  padding: 26px 30px 24px 30px;
  background:
    radial-gradient(600px 240px at 88% 10%, rgba(34, 201, 230, 0.35), transparent 65%),
    radial-gradient(500px 220px at 10% 110%, rgba(215, 38, 61, 0.35), transparent 60%),
    linear-gradient(115deg, #05101f 0%, #0a1a33 42%, #123b74 100%);
  color: #ffffff;
  box-shadow: 0 18px 40px -22px rgba(6, 24, 51, 0.75);
  border: 1px solid rgba(34, 201, 230, 0.25);
}
.eb-hero__eyebrow {
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #8fe6f6;
  font-weight: 700;
  margin-bottom: 6px;
}
.eb-hero__title {
  font-size: 2.5rem;
  line-height: 1.05;
  font-weight: 800;
  margin: 0 0 6px 0;
  letter-spacing: -0.02em;
}
.eb-hero__title span { color: #4fd7ef; }
.eb-hero__subtitle {
  font-size: 1.02rem;
  color: #d7e6f7;
  margin: 0 0 12px 0;
  max-width: 860px;
}
.eb-hero__chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.eb-chip {
  display: inline-block;
  border: 1px solid rgba(143, 230, 246, 0.45);
  background: rgba(9, 32, 62, 0.55);
  color: #cfeeff;
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.74rem;
  letter-spacing: 0.04em;
}
.eb-chip--accent { border-color: rgba(215, 38, 61, 0.65); color: #ffd7dd; }
.eb-hero__motto {
  margin-top: 14px;
  font-size: 0.78rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #9fc7f0;
}
.eb-hero__motto b { color: #ffffff; }

/* ---------- section headings ---------- */
.eb-section {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 26px 0 12px 0;
}
.eb-section__num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px; height: 32px;
  border-radius: 10px;
  background: linear-gradient(160deg, var(--eb-red), #a8122b);
  color: #fff;
  font-weight: 800;
  font-size: 0.95rem;
  box-shadow: 0 8px 18px -10px rgba(215, 38, 61, 0.9);
}
.eb-section__title {
  font-size: 1.16rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  color: var(--eb-navy);
  text-transform: uppercase;
}
.eb-section__rule {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--eb-line), transparent);
}

/* ---------- status cards ---------- */
.eb-status {
  border-radius: 16px;
  padding: 18px 20px;
  border: 1px solid var(--eb-line);
  background: #ffffff;
  box-shadow: 0 10px 26px -22px rgba(10, 26, 51, 0.6);
}
.eb-status--answerable { border-left: 6px solid var(--eb-green); background: linear-gradient(90deg, var(--eb-green-soft), #ffffff 55%); }
.eb-status--partial { border-left: 6px solid var(--eb-orange); background: linear-gradient(90deg, var(--eb-orange-soft), #ffffff 55%); }
.eb-status--abstain { border-left: 6px solid var(--eb-purple); background: linear-gradient(90deg, var(--eb-purple-soft), #ffffff 55%); }
.eb-status__head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.eb-status__icon { font-size: 1.3rem; line-height: 1; }
.eb-status__label {
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.8rem;
}
.eb-status--answerable .eb-status__label { color: var(--eb-green); }
.eb-status--partial .eb-status__label { color: var(--eb-orange); }
.eb-status--abstain .eb-status__label { color: var(--eb-purple); }
.eb-status__body { color: var(--eb-ink); font-size: 0.95rem; }
.eb-abstain-message {
  font-size: 1.06rem;
  font-weight: 700;
  color: var(--eb-purple);
  background: #ffffff;
  border: 1px dashed rgba(107, 33, 168, 0.45);
  border-radius: 12px;
  padding: 12px 14px;
  margin: 8px 0 4px 0;
}

/* ---------- answer ---------- */
.eb-answer {
  background: #ffffff;
  border: 1px solid var(--eb-line);
  border-radius: 16px;
  padding: 20px 22px;
  font-size: 1rem;
  line-height: 1.62;
  color: var(--eb-ink);
  box-shadow: 0 10px 26px -24px rgba(10, 26, 51, 0.7);
}
.eb-answer p { margin: 0 0 10px 0; }

/* ---------- metric tiles ---------- */
.eb-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 10px; }
.eb-metric {
  background: #ffffff;
  border: 1px solid var(--eb-line);
  border-radius: 14px;
  padding: 12px 14px;
  border-top: 3px solid var(--eb-cyan);
}
.eb-metric__label {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--eb-muted);
  font-weight: 700;
}
.eb-metric__value { font-size: 1.5rem; font-weight: 800; color: var(--eb-navy); line-height: 1.15; }
.eb-metric__hint { font-size: 0.72rem; color: var(--eb-muted); }

/* ---------- citation cards ---------- */
.eb-cite {
  border: 1px solid var(--eb-line);
  border-left: 5px solid var(--eb-cyan);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 10px;
  background: #ffffff;
}
.eb-cite__top { display: flex; gap: 10px; align-items: flex-start; }
.eb-cite__num {
  flex: 0 0 auto;
  min-width: 30px; height: 30px;
  border-radius: 8px;
  background: var(--eb-navy);
  color: #fff;
  font-weight: 800;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.86rem;
}
.eb-cite__title { font-weight: 700; color: var(--eb-navy); font-size: 0.98rem; }
.eb-cite__meta { color: var(--eb-muted); font-size: 0.82rem; margin-top: 2px; }
.eb-badge {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-radius: 999px;
  padding: 2px 9px;
  background: var(--eb-cyan-soft);
  color: #0b6f86;
  border: 1px solid rgba(34, 201, 230, 0.5);
}
.eb-badge--warn { background: var(--eb-orange-soft); color: var(--eb-orange); border-color: rgba(180, 83, 9, 0.35); }
.eb-badge--info { background: #eef3fb; color: #24487f; border-color: #c9d8ee; }
.eb-badge--danger { background: #fdecef; color: #a5122b; border-color: rgba(215, 38, 61, 0.35); }
.eb-quote {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #f8fbfe;
  border: 1px solid var(--eb-line);
  color: #22344d;
  font-size: 0.88rem;
  line-height: 1.55;
}
.eb-quote__label {
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--eb-muted);
  font-weight: 700;
  margin-bottom: 4px;
}

/* ---------- strength ---------- */
.eb-strength {
  border-radius: 16px;
  padding: 16px 18px;
  border: 1px solid var(--eb-line);
  background: #ffffff;
}
.eb-strength__label { font-weight: 800; font-size: 1.02rem; }
.eb-strength__score { color: var(--eb-muted); font-size: 0.8rem; }
.eb-bar { height: 8px; border-radius: 999px; background: #eef2f8; overflow: hidden; margin: 8px 0 4px 0; }
.eb-bar__fill { height: 100%; border-radius: 999px; }

/* ---------- notices ---------- */
.eb-notice {
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 0.88rem;
  border: 1px solid var(--eb-line);
  background: #fff;
  margin-bottom: 8px;
}
.eb-notice--warn { border-left: 4px solid var(--eb-orange); background: var(--eb-orange-soft); color: #6b3f06; }
.eb-notice--danger { border-left: 4px solid var(--eb-red); background: #fdecef; color: #82101f; }
.eb-notice--info { border-left: 4px solid var(--eb-blue); background: #eef4fd; color: #1b3a6b; }
.eb-notice--ok { border-left: 4px solid var(--eb-green); background: var(--eb-green-soft); color: #0d5c43; }

/* ---------- misc ---------- */
.eb-small { color: var(--eb-muted); font-size: 0.8rem; }
.eb-kv { display: grid; grid-template-columns: 140px 1fr; gap: 4px 10px; font-size: 0.85rem; }
.eb-kv dt { color: var(--eb-muted); font-weight: 600; }
.eb-kv dd { margin: 0; color: var(--eb-ink); }
.eb-footer {
  margin-top: 34px;
  padding: 16px 18px;
  border-radius: 14px;
  background: linear-gradient(110deg, #0a1a33, #123b74);
  color: #cfe4f8;
  text-align: center;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.76rem;
}
.eb-footer b { color: #ffffff; }

/* Streamlit widget cosmetics */
.stButton > button {
  border-radius: 10px;
  border: 1px solid #c9d7ea;
  font-weight: 600;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(160deg, #14479c, #0a1a33);
  border: none;
  color: #ffffff;
  font-weight: 700;
}
.stButton > button[kind="primary"]:hover { background: linear-gradient(160deg, #1d5fd6, #123b74); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f3f8fd, #ffffff 40%);
  border-right: 1px solid var(--eb-line);
}
[data-testid="stSidebar"] h2 { font-size: 1.02rem; color: var(--eb-navy); }
div[data-testid="stExpander"] {
  border: 1px solid var(--eb-line);
  border-radius: 12px;
  background: #ffffff;
}
div[data-testid="stExpander"] summary { font-weight: 600; color: var(--eb-navy); }
.stTextInput input { border-radius: 10px; }
</style>
"""
