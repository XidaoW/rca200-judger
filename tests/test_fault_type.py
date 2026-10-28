"""M2 fixture + synthetic tests: rcaeval.fault_type must reproduce brise F pipeline.

Coverage by tier:
  EXACT       — live fixture e_mq (F=1.0 expected)
  CLUSTER     — synthetic, uses real fault_causal_clusters.json
  CAUSAL_KHOP — synthetic, walks a GT reasoning_steps chain
  LLM rescue  — stub LLM returning canned causal_downstream_1hop result
  enum penalty — synthetic, multiple actuals on CLUSTER tier
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brise_rca_score.llm.stub import StubLLM
from brise_rca_score.scoring.fault_type import (
    ReasoningStep,
    load_causal_clusters,
    score_fault_type,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())["data"]


# ---------------------------------------------------------------------------
# EXACT — locked against brise e_mq fixture (F=1.0)
# ---------------------------------------------------------------------------
def test_fault_type_exact_from_mq_fixture():
    data = _load("e_mq.json")
    gt = data["ground_truth"]["expected_fault_id"]  # "F005-messageQueueBacklog"
    agent = data["diagnosis"]["fault_candidates"]
    actual = [c["fault_id"] for c in agent]  # ["F005-messageQueueBacklog"]
    brise_f = data["scores"]["fault_type"]
    assert brise_f == 1.0, f"fixture drifted: brise F={brise_f}"

    ev = score_fault_type(
        expected_fault_type=gt,
        actual_fault_types=actual,
    )
    assert ev.score == 1.0
    assert ev.match_tier == "exact"
    assert ev.scoring_method == "exact"
    assert ev.matched_fault_id == "F005-messageQueueBacklog"
    assert ev.predicted_fault_ids == ["F005-messageQueueBacklog"]


def test_fault_type_exact_case_insensitive():
    """Brise normalizes both sides to lowercase before comparing."""
    ev = score_fault_type(
        expected_fault_type="F010-slowSQL",
        actual_fault_types=["f010-SLOWSQL"],
    )
    assert ev.score == 1.0
    assert ev.match_tier == "exact"


# ---------------------------------------------------------------------------
# CLUSTER — fault_causal_clusters.json
# ---------------------------------------------------------------------------
def test_fault_type_cluster_match():
    """F022-fullGC and F007-memoryPressure are in 'JVM 内存簇' (refreshed 2026-06-06).

    Verified against shipped data/fault_causal_clusters.json.
    """
    cfg = load_causal_clusters()
    cluster = cfg.fault_id_to_cluster.get("f022-fullgc")
    assert (
        cluster == "jvm_内存簇"
    ), f"shipped fault_causal_clusters.json drifted; f022-fullgc → {cluster}"

    ev = score_fault_type(
        expected_fault_type="F007-memoryPressure",
        actual_fault_types=["F022-fullGC"],
    )
    assert ev.score == 0.5
    assert ev.match_tier == "cluster"
    assert ev.scoring_method == "cluster"
    assert ev.shared_category == "JVM 内存簇"
    assert "同簇匹配" in ev.match_detail


def test_fault_type_cluster_does_not_match_across_clusters():
    """F022-fullGC (perf cluster) vs F029-redisUnavailable (dependency cluster) → NONE."""
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F029-redisUnavailable"],
        # Provide a stub LLM so we don't skip rescue — but the stub will return
        # malformed JSON so rescue fails and we get NONE.
    )
    assert ev.score == 0.0
    assert ev.match_tier == "none"


# ---------------------------------------------------------------------------
# GT causal chain — score follows hop distance
# ---------------------------------------------------------------------------
def test_fault_type_gt_chain_1hop():
    """GT root step is F002-threadExhaustion (step=1=EXACT), step=2 is F014-httpError5xx.

    Agent says F014 → should hit step 2 → CAUSAL_1HOP score 0.85.
    """
    gt_chain = [
        ReasoningStep(step=1, target="inventory", fault_id="F002-threadExhaustion"),
        ReasoningStep(step=2, target="cart", fault_id="F014-httpError5xx"),
    ]
    ev = score_fault_type(
        expected_fault_type="F002-threadExhaustion",
        actual_fault_types=["F014-httpError5xx"],
        gt_causal_chain=gt_chain,
    )
    assert ev.score == 0.85, f"expected 1HOP=0.85, got {ev.score}"
    assert ev.match_tier == "causal_1hop"
    assert ev.causal_match_depth == 1


def test_fault_type_gt_chain_3hop_floor():
    """step >= 4 clamps to CAUSAL_3HOP (score 0.50)."""
    gt_chain = [
        ReasoningStep(step=1, target="inv", fault_id="F002-threadExhaustion"),
        ReasoningStep(step=2, target="cart", fault_id="F014-httpError5xx"),
        ReasoningStep(step=3, target="frontend", fault_id="F011-codeDefect"),
        ReasoningStep(step=5, target="api", fault_id="F099-deepSymptom"),
    ]
    ev = score_fault_type(
        expected_fault_type="F002-threadExhaustion",
        actual_fault_types=["F099-deepSymptom"],
        gt_causal_chain=gt_chain,
    )
    assert ev.score == 0.50
    assert ev.match_tier == "causal_3hop"


# ---------------------------------------------------------------------------
# LLM rescue — stub returns canned JSON
# ---------------------------------------------------------------------------
def _make_rescue_response(
    relation: str, best_actual: str, confidence: float = 0.85
) -> str:
    return json.dumps(
        {
            "steps": {
                "chain_hits": [],
                "direct_related": [best_actual]
                if relation == "causal_downstream_1hop"
                else [],
                "indirect_related": [],
                "unrelated": [],
            },
            "best_actual": best_actual,
            "relation": relation,
            "confidence": confidence,
            "chain": f"模拟 {relation} 推理",
            "chain_path": [
                {"target": "x", "fault_id": "y", "role": "mechanism", "source": "llm"}
            ],
            "chain_direction": "downstream" if "downstream" in relation else "upstream",
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "relation,expected_score,expected_tier",
    [
        ("causal_downstream_1hop", 0.50, "llm_causal_downstream_1hop"),
        ("causal_downstream_2hop", 0.35, "llm_causal_downstream_2hop"),
        ("causal_upstream", 0.50, "llm_causal_upstream"),
        ("unrelated", 0.00, "none"),
    ],
)
def test_fault_type_llm_rescue(relation, expected_score, expected_tier):
    """LLM rescue maps relation_type → score per brise's score map."""
    stub = StubLLM(_make_rescue_response(relation, "F014-httpError5xx"))
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",  # not in any cluster with F099-other
        actual_fault_types=["F099-other"],  # outside cluster + dictionary
        llm=stub,
        agent_chain_text="(mock)",
    )
    assert (
        ev.score == expected_score
    ), f"{relation}: expected score {expected_score}, got {ev.score}"
    if relation == "unrelated":
        assert ev.match_tier == "none"
        assert ev.scoring_method == "unmatched"
    else:
        assert ev.match_tier == expected_tier
        assert ev.scoring_method == "llm_causal_rescue"


def test_fault_type_llm_rescue_low_confidence_rejected():
    """confidence < 0.4 → positive relation rejected, falls back to NONE."""
    stub = StubLLM(
        _make_rescue_response(
            "causal_downstream_1hop", "F014-httpError5xx", confidence=0.3
        )
    )
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-other"],
        llm=stub,
    )
    assert ev.score == 0.0
    assert ev.match_tier == "none"


def test_fault_type_causal_judge_rejects_malformed_response():
    stub = StubLLM("{}")
    with pytest.raises(ValueError, match="invalid response"):
        score_fault_type(
            expected_fault_type="F022-fullGC",
            actual_fault_types=["F099-other"],
            llm=stub,
        )
    assert len(stub.calls) == 3
    assert "上一次响应未满足结构化协议" in stub.calls[1][0].content


def test_fault_type_causal_judge_uses_structured_retry_capability():
    final = _make_rescue_response("causal_upstream", "F014-httpError5xx")

    class RetryAwareLLM:
        normal_calls = 0
        retry_calls = 0

        def chat(self, *args, **kwargs):
            self.normal_calls += 1
            return "{}"

        def chat_structured_retry(self, *args, **kwargs):
            self.retry_calls += 1
            return final

    llm = RetryAwareLLM()
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-other"],
        llm=llm,
    )
    assert ev.score == 0.50
    assert llm.normal_calls == 1
    assert llm.retry_calls == 1


def test_fault_type_causal_judge_accepts_json_after_reasoning_braces():
    """A reasoning preamble with braces must not swallow the final JSON object."""
    final = _make_rescue_response("causal_downstream_1hop", "F014-httpError5xx")
    raw = 'analysis example: {"draft": true}\n```json\n' + final + "\n```"
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-other"],
        llm=StubLLM(raw),
        agent_chain_text="(mock)",
    )
    assert ev.score == 0.50
    assert ev.scoring_method == "llm_causal_rescue"


def test_fault_type_causal_judge_skips_schema_example_before_final_json():
    """A decodable schema/example object is ignored unless relation is valid."""
    final = _make_rescue_response("causal_upstream", "F014-httpError5xx")
    raw = '{"relation": "one of the allowed values"}\n' + final
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-other"],
        llm=StubLLM(raw),
    )
    assert ev.score == 0.50
    assert ev.match_tier == "llm_causal_upstream"


def test_fault_type_causal_judge_rejects_invalid_confidence_then_uses_final_json():
    final = _make_rescue_response("unrelated", "F014-httpError5xx")
    raw = '{"relation":"causal_upstream","confidence":"high"}\n' + final
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-other"],
        llm=StubLLM(raw),
    )
    assert ev.score == 0.0
    assert ev.scoring_method == "unmatched"


def test_fault_type_no_llm_means_no_rescue():
    """When llm is None, score stays 0 with no rescue attempt."""
    ev = score_fault_type(
        expected_fault_type="F022-fullGC",
        actual_fault_types=["F099-totally-unrelated"],
        llm=None,
    )
    assert ev.score == 0.0
    assert ev.match_tier == "none"
    assert ev.scoring_method == "unmatched"


# ---------------------------------------------------------------------------
# Enumeration penalty
# ---------------------------------------------------------------------------
def test_fault_type_no_enum_penalty_on_cluster():
    """Brise does NOT apply enum penalty to CLUSTER tier (early return at line 191
    before _apply_enumeration_penalty at line 264 in brise fault_type_scorer.py).

    Pair from refreshed clusters: F007-memoryPressure + F022-fullGC ∈ "JVM 内存簇".
    Multiple candidates → still score=0.5 (no 1/√n penalty).
    """
    ev = score_fault_type(
        expected_fault_type="F007-memoryPressure",
        actual_fault_types=[
            "F022-fullGC",
            "F099-irrelevant",
            "F098-other",
            "F097-noise",
        ],
    )
    assert ev.score == 0.5, f"expected 0.5 (no penalty on CLUSTER), got {ev.score}"
    assert ev.match_tier == "cluster"


def test_fault_type_enumeration_penalty_NOT_applied_on_exact():
    """EXACT bypass: enum penalty doesn't apply even with many candidates."""
    ev = score_fault_type(
        expected_fault_type="F010-slowSQL",
        actual_fault_types=["F010-slowSQL", "F099-other", "F098-also-other"],
    )
    assert ev.score == 1.0


# ---------------------------------------------------------------------------
# Empty / edge
# ---------------------------------------------------------------------------
def test_fault_type_empty_expected_returns_zero():
    ev = score_fault_type(expected_fault_type="", actual_fault_types=["F010-slowSQL"])
    assert ev.score == 0.0
    assert ev.match_tier == "none"


def test_fault_type_empty_actual_returns_zero():
    ev = score_fault_type(expected_fault_type="F010-slowSQL", actual_fault_types=[])
    assert ev.score == 0.0
    assert ev.match_tier == "none"
