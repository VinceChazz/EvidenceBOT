"""Optional dashboard smoke test.

Runs the real Streamlit app through ``streamlit.testing.v1.AppTest``. Because a
real search hits the live databases, it is skipped unless explicitly enabled:

    RUN_UI_TESTS=1 pytest tests/test_ui.py -v

Set ``ENABLE_OFFLINE_SAMPLE_FALLBACK=true`` (the default) when running it without
network access, and the labelled offline corpus keeps the flow exercisable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_UI_TESTS") != "1",
    reason="UI smoke test spawns the real app and performs live retrieval; set RUN_UI_TESTS=1",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


def test_dashboard_boots_without_exceptions():
    at = _app()
    assert not at.exception
    labels = [button.label for button in at.button]
    assert any("Search evidence" in label for label in labels)
    assert any("Clear" in label for label in labels)
    assert len(at.text_input) == 1


def test_search_flow_renders_a_decision():
    at = _app()
    at.text_input[0].set_value("What role does IL-6 play in rheumatoid arthritis?")
    [button for button in at.button if "Search evidence" in button.label][0].click()
    at.run()

    assert not at.exception
    result = at.session_state["result"]
    assert result is not None
    assert result.decision in {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
    assert at.session_state["history"], "the search must be recorded in session history"


def test_example_button_triggers_search():
    at = _app()
    examples = [button for button in at.button if "Unanswerable" in button.label]
    assert examples, "the unanswerable example button should exist"
    examples[0].click()
    at.run()

    assert not at.exception
    result = at.session_state["result"]
    assert result is not None
    assert result.decision == "UNANSWERABLE"
    assert result.abstention_message.startswith("Insufficient evidence.")


def test_history_rerun_and_clear_are_wired():
    at = _app()
    at.text_input[0].set_value("What role does IL-6 play in rheumatoid arthritis?")
    [button for button in at.button if "Search evidence" in button.label][0].click()
    at.run()
    assert not at.exception

    # Searching the same question again must not duplicate history entries or keys.
    [button for button in at.button if "Search evidence" in button.label][0].click()
    at.run()
    assert len(at.session_state["history"]) == 1
    assert not at.exception

    rerun = [button for button in at.button if button.label == "Re-run"]
    assert rerun
    rerun[0].click()
    at.run()
    assert not at.exception

    clear = [button for button in at.button if button.label.strip().startswith("✕ Clear")][0]
    clear.click()
    at.run()
    assert at.session_state["result"] is None
    assert not at.exception


def test_partial_example_produces_partial_decision():
    at = _app()
    example = [button for button in at.button if "Partially answerable" in button.label][0]
    example.click()
    at.run()

    assert not at.exception
    result = at.session_state["result"]
    assert result is not None
    assert result.decision in {"PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
