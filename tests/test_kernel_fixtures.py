"""Fixture tests: rcaeval.kernel must reproduce brise overall_score exactly.

Each fixture is a real brise `POST /api/v1/eval/rca-drill/evaluate` response
captured 2026-06-05 against trace_ids from Aone run 44241127.

Expected behavior:
    Feed each fixture's per-dimension scores (F, E, P) + entity_grade +
    extraction_method + credibility into rcaeval.compute_weighted_result.
    Output `overall_score` must equal brise's reported `scores.overall`.

This locks the M0 kernel against the production scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brise_rca_score.scoring.kernel import compute_weighted_result
from brise_rca_score.scoring.models import (
    CredibilityScoreEvidence,
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    ProcessScoreEvidence,
    ScoreWeights,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(path: str) -> dict:
    return json.loads((FIXTURE_DIR / path).read_text())["data"]


def _evidence_from_brise(data: dict):
    """Build the 4 evidence dataclasses from brise's evaluate response."""
    s = data["scores"]
    d = data["details"]

    fault = FaultTypeScoreEvidence(
        score=s["fault_type"],
        match_tier=d["fault_type"].get("match_tier", ""),
        matched_fault_id=d["fault_type"].get("matched_fault_id", ""),
        predicted_fault_ids=d["fault_type"].get("predicted_fault_ids", []),
        shared_category=d["fault_type"].get("shared_category", ""),
        scoring_method=d["fault_type"].get("scoring_method", ""),
        reason=d["fault_type"].get("reason", ""),
        match_detail=d["fault_type"].get("match_detail", ""),
    )

    # extraction_method: brise's entity detail carries the extractor's status.
    # The cause-entity penalty only fires when E missed AND extraction != llm_failed.
    diag_status = data.get("diagnosis", {}).get("extraction_status", "")
    extraction_method = "llm_failed" if diag_status == "llm_failed" else ""

    entity = EntityScoreEvidence(
        score=s["entity"],
        entity_grade=d["entity"].get("entity_grade", ""),
        entity_grade_score=d["entity"].get("score", 0.0) * 100,
        expected_count=d["entity"].get("expected_count", 0),
        matched_count=d["entity"].get("matched_count", 0),
        matched_entities=d["entity"].get("matched_entities", []),
        entity_resolve_method=d["entity"].get("entity_resolve_method", ""),
        reason=d["entity"].get("reason", ""),
        extraction_method=extraction_method,
    )

    process = ProcessScoreEvidence(
        score=s["process"],
        score_source=d["process"].get("score_source", ""),
        reason=d["process"].get("reason", ""),
        review_dimensions=d["process"].get("review_dimensions", []),
    )

    # Credibility: brise's evaluate response doesn't expose conclusion_reliability
    # at top level in these 4 fixtures (all are RELIABLE in production). Default
    # to RELIABLE = gate × 1.0. If a future fixture hits QUESTIONABLE / UNRELIABLE
    # we will surface it from data["credibility"] (TBD field path).
    credibility = CredibilityScoreEvidence(conclusion_reliability="RELIABLE")

    return fault, entity, process, credibility


@pytest.mark.parametrize(
    "fixture,expected_overall",
    [
        ("e_slowSQL.json", 64.0),  # F=0, E=1, P=0.8, no penalty → 100×(0+0.4+0.24)=64
        (
            "e_mq.json",
            33.0,
        ),  # F=1, E=0 (未命中), P=0.5, penalty=0.6 → 100×(0.18+0+0.15)=33
        ("e_hotspot.json", 55.0),  # F=0, E=1, P=0.5, no penalty → 100×(0+0.4+0.15)=55
        ("eval_resp.json", 0.0),  # extraction_failed → all zero, skip_evaluate gate
    ],
)
def test_overall_matches_brise(fixture, expected_overall):
    data = _load(fixture)
    brise_reported = data["scores"]["overall"]
    assert brise_reported == expected_overall, (
        f"fixture {fixture} reported brise overall={brise_reported}, "
        f"expected {expected_overall}"
    )

    fault, entity, process, credibility = _evidence_from_brise(data)
    result = compute_weighted_result(
        fault_evidence=fault,
        entity_evidence=entity,
        process_evidence=process,
        credibility_evidence=credibility,
    )
    assert result.overall_score == expected_overall, (
        f"rcaeval overall={result.overall_score} ≠ brise {expected_overall} "
        f"(formula: {result.overall_formula})"
    )


def test_entity_penalty_triggers_only_when_missed():
    """Cause-entity penalty (0.6×) fires iff E_grade == '未命中' and extraction != 'llm_failed'."""
    # Construct a synthetic F=1, E=0 (未命中) scenario, no LLM extraction failure
    fault = FaultTypeScoreEvidence(score=1.0)
    entity_missed = EntityScoreEvidence(
        score=0.0, entity_grade="未命中", extraction_method=""
    )
    entity_llmfail = EntityScoreEvidence(
        score=0.0, entity_grade="未命中", extraction_method="llm_failed"
    )
    process = ProcessScoreEvidence(score=0.0)
    credibility = CredibilityScoreEvidence(conclusion_reliability="RELIABLE")

    # Penalty case: F-term ×0.6 → 100×(1×0.6×0.3 + 0 + 0) = 18
    r1 = compute_weighted_result(
        fault_evidence=fault,
        entity_evidence=entity_missed,
        process_evidence=process,
        credibility_evidence=credibility,
    )
    assert r1.overall_score == 18.0, f"expected 18.0, got {r1.overall_score}"

    # llm_failed case: penalty disabled → 100×(1×1.0×0.3 + 0 + 0) = 30
    r2 = compute_weighted_result(
        fault_evidence=fault,
        entity_evidence=entity_llmfail,
        process_evidence=process,
        credibility_evidence=credibility,
    )
    assert r2.overall_score == 30.0, f"expected 30.0, got {r2.overall_score}"


def test_credibility_gate():
    """Overall uses current Brise multipliers 1.0 / 0.5 / 0.2."""
    fault = FaultTypeScoreEvidence(score=1.0)
    entity = EntityScoreEvidence(score=1.0, entity_grade="精确命中")
    process = ProcessScoreEvidence(score=1.0)
    for level, expected in [
        ("RELIABLE", 100.0),
        ("QUESTIONABLE", 50.0),
        ("UNRELIABLE", 20.0),
    ]:
        cred = CredibilityScoreEvidence(conclusion_reliability=level)
        result = compute_weighted_result(
            fault_evidence=fault,
            entity_evidence=entity,
            process_evidence=process,
            credibility_evidence=cred,
        )
        assert result.overall_score == expected


def test_process_na_rebalances_weights():
    result = compute_weighted_result(
        fault_evidence=FaultTypeScoreEvidence(score=1.0),
        entity_evidence=EntityScoreEvidence(score=0.0, entity_grade="未命中"),
        process_evidence=ProcessScoreEvidence(
            score=0.0,
            score_source="rca_judge_failed",
        ),
        credibility_evidence=CredibilityScoreEvidence(),
    )
    assert result.weights.process == 0.0
    assert result.overall_score == 25.7


def test_zero_weights_fail_explicitly():
    with pytest.raises(ValueError, match="weights sum to 0"):
        compute_weighted_result(
            fault_evidence=FaultTypeScoreEvidence(score=1.0),
            entity_evidence=EntityScoreEvidence(score=1.0),
            process_evidence=ProcessScoreEvidence(score=1.0),
            credibility_evidence=CredibilityScoreEvidence(),
            weights=ScoreWeights(
                fault_type_weight=0.0,
                entity_weight=0.0,
                process_weight=0.0,
            ),
        )
