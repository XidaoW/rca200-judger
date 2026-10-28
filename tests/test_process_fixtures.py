"""M1 fixture tests: rcaeval.process must reproduce brise's process score.

Two paths verified against the same 3 brise samples that have non-zero P:

  Path A (offline):
    Feed `details.process.review_dimensions` directly to
    score_process_from_review_dimensions, assert P matches brise.

  Path B (LLM path with stub):
    Build a StubLLM that returns the SAME review_dimensions[] (as a JSON
    string mimicking what a real LLM would return), call score_process,
    assert P matches brise. This locks the prompt-pack → JSON-parse →
    _build_review_evidence pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brise_rca_score.llm.stub import StubLLM
from brise_rca_score.scoring.process import (
    _score_review_dimension,
    score_process,
    score_process_from_review_dimensions,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())["data"]


@pytest.mark.parametrize(
    "fixture,expected_p",
    [
        ("e_slowSQL.json", 0.8),
        ("e_mq.json", 0.5),
        ("e_hotspot.json", 0.5),
    ],
)
def test_process_offline_matches_brise(fixture, expected_p):
    data = _load(fixture)
    brise_dims = data["details"]["process"]["review_dimensions"]
    brise_p = data["scores"]["process"]
    assert brise_p == expected_p, f"fixture {fixture} brise P drifted: {brise_p}"

    ev = score_process_from_review_dimensions(
        brise_dims,
        expected_fault_type=data["ground_truth"]["expected_fault_id"],
    )
    assert (
        ev.score == expected_p
    ), f"rcaeval P={ev.score} ≠ brise {expected_p} for {fixture}"
    # also lock the explanation
    assert ev.explanation_for_human == f"过程 {round(expected_p * 100)}/100（三维审查）"
    assert ev.score_source == "rca_review"


@pytest.mark.parametrize(
    "fixture,expected_p",
    [
        ("e_slowSQL.json", 0.8),
        ("e_mq.json", 0.5),
        ("e_hotspot.json", 0.5),
    ],
)
def test_process_via_stub_llm(fixture, expected_p):
    """End-to-end through score_process with a stub LLM returning brise dims."""
    data = _load(fixture)
    brise_dims = data["details"]["process"]["review_dimensions"]
    canned = json.dumps({"dimensions": brise_dims, "reasoning": ""}, ensure_ascii=False)
    stub = StubLLM(canned)

    ev = score_process(
        llm=stub,
        analysis_output="(fixture replay — analysis text omitted)",
        expected_fault_type=data["ground_truth"]["expected_fault_id"],
        injection_target=data["ground_truth"].get("target_entity_names", [""])[0],
        causal_chain_text="(omitted in fixture)",
    )
    assert ev.score == expected_p
    assert len(ev.review_dimensions) == 3
    assert len(stub.calls) == 1


def test_score_review_dimension_table():
    """Lock the per-dimension severity table (brise rca_process_scorer.py:506-522)."""
    assert _score_review_dimension([]) == 100
    assert _score_review_dimension([{"severity": "low"}]) == 70
    assert _score_review_dimension([{"severity": "low"}, {"severity": "low"}]) == 70
    assert _score_review_dimension([{"severity": "high"}]) == 50
    assert _score_review_dimension([{"severity": "high"}, {"severity": "high"}]) == 20
    assert _score_review_dimension([{"severity": "high"}] * 3) == 0
    # mixed
    assert _score_review_dimension([{"severity": "high"}, {"severity": "low"}]) == 50
    assert _score_review_dimension([{"severity": "high"}] * 4) == 0


def test_process_skipped_on_empty_analysis():
    """Brise rule: empty analysis_output returns rca_no_analysis evidence."""
    stub = StubLLM("{}")
    ev = score_process(
        llm=stub,
        analysis_output="   ",
        expected_fault_type="F002-threadExhaustion",
    )
    assert ev.score == 0.0
    assert ev.score_source == "skipped"
    assert len(stub.calls) == 0  # LLM should not be invoked


def test_process_handles_markdown_fence():
    """LLMs commonly wrap JSON in ```json fences — must be stripped."""
    dims = [
        {"dimension": "causal_contradiction", "issues": []},
        {"dimension": "logic_jump", "issues": []},
        {"dimension": "evidence_gap", "issues": []},
    ]
    canned = "```json\n" + json.dumps({"dimensions": dims, "reasoning": "ok"}) + "\n```"
    ev = score_process(
        llm=StubLLM(canned),
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
    )
    assert ev.score == 1.0  # all dims clean → 100/100/100/100 → 1.0


def test_process_handles_reasoning_preamble_with_braces():
    dims = [
        {"dimension": "causal_contradiction", "issues": []},
        {"dimension": "logic_jump", "issues": []},
        {"dimension": "evidence_gap", "issues": []},
    ]
    final = json.dumps({"dimensions": dims, "reasoning": "ok"})
    raw = 'analysis example: {"draft": true}\n```json\n' + final + "\n```"
    ev = score_process(
        llm=StubLLM(raw),
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
    )
    assert ev.score == 1.0


def test_process_skips_incomplete_schema_example_before_final_json():
    dims = [
        {"dimension": "causal_contradiction", "issues": []},
        {"dimension": "logic_jump", "issues": []},
        {"dimension": "evidence_gap", "issues": []},
    ]
    example = json.dumps({"dimensions": [{"dimension": "example", "issues": []}]})
    final = json.dumps({"dimensions": dims, "reasoning": "ok"})
    ev = score_process(
        llm=StubLLM(example + "\n" + final),
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
    )
    assert ev.score == 1.0


def test_process_rejects_response_without_any_json_object():
    with pytest.raises(ValueError, match="no decodable object"):
        score_process(
            llm=StubLLM("analysis only; no final object"),
            analysis_output="x",
            expected_fault_type="F010-slowSQL",
        )


def test_process_rejects_incomplete_dimensions():
    canned = json.dumps(
        {
            "dimensions": [{"dimension": "causal_contradiction", "issues": []}],
            "reasoning": "incomplete",
        }
    )
    with pytest.raises(ValueError, match="exactly three unique dimensions"):
        score_process(
            llm=StubLLM(canned),
            analysis_output="x",
            expected_fault_type="F010-slowSQL",
        )


def test_process_accepts_retry_instruction_without_relaxing_schema():
    dims = [
        {"dimension": "causal_contradiction", "issues": []},
        {"dimension": "logic_jump", "issues": []},
        {"dimension": "evidence_gap", "issues": []},
    ]
    ev = score_process(
        llm=StubLLM(json.dumps({"dimensions": dims, "reasoning": "ok"})),
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
        retry_instruction="return the exact three-dimension schema",
    )
    assert ev.score == 1.0


def test_process_uses_structured_retry_capability_when_requested():
    dims = [
        {"dimension": "causal_contradiction", "issues": []},
        {"dimension": "logic_jump", "issues": []},
        {"dimension": "evidence_gap", "issues": []},
    ]

    class RetryAwareLLM:
        retry_called = False

        def chat(self, *args, **kwargs):
            raise AssertionError("normal chat must not be used")

        def chat_structured_retry(self, *args, **kwargs):
            self.retry_called = True
            return json.dumps({"dimensions": dims, "reasoning": "ok"})

    llm = RetryAwareLLM()
    ev = score_process(
        llm=llm,
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
        structured_retry=True,
    )
    assert ev.score == 1.0
    assert llm.retry_called is True


def test_process_normalizes_dict_dimensions_and_alias_fields():
    """Current judge models may return dimensions as a mapping with alias fields."""
    canned = json.dumps(
        {
            "dimensions": {
                "causal_contradiction": {
                    "strengths": [{"content": "direction is correct"}],
                    "issues": [{"severity": "high", "title": "mechanism mismatch"}],
                },
                "logic_jump": {"strengths": [], "issues": []},
                "evidence_gap": {
                    "strengths": [],
                    "issues": [{"severity": "low", "content": "missing baseline"}],
                },
            },
            "reasoning": "normalized response",
        }
    )
    ev = score_process(
        llm=StubLLM(canned),
        analysis_output="x",
        expected_fault_type="F010-slowSQL",
    )
    assert ev.score == pytest.approx(0.7333)
    assert ev.review_dimensions[0]["issues"][0]["description"] == "mechanism mismatch"
