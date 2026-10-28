"""rcaeval.scoring.kernel — O (overall) kernel + cause-entity weight + credibility gate.

Ported verbatim from cms2.0-route-agent/vibeops/eval/rca_evaluate/engine/scoring/
composite_scorer.py:754-866 (functions _compute_cause_entity_weight and
_compute_weighted_result), with the only change being:

  - ConfigManager.get_config("CAUSE_ENTITY_WEIGHT_ENABLED", "true")
    →  ScoringConfig.cause_entity_weight_enabled (dataclass field, default True)

Formula (validated against 4 real brise samples, see tests/test_kernel_fixtures.py):

    entity_penalty   = 0.6 if (E_grade == "未命中" and extraction_method != "llm_failed")
                       else 1.0
    raw_composite    = 100 × (F × entity_penalty × w_F + E × w_E + P × w_P)
    credibility_gate = {UNRELIABLE: 0.2, QUESTIONABLE: 0.5, RELIABLE: 1.0}
    overall          = round(raw_composite × credibility_gate, 1)

Default weights are w_F = 0.3, w_E = 0.4, w_P = 0.3. Evaluator-side process
NA removes P from the denominator and renormalizes F/E.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    CredibilityScoreEvidence,
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    NormalizedWeights,
    ProcessScoreEvidence,
    RcaScoreResult,
    ScoreWeights,
)


@dataclass
class ScoringConfig:
    """Open-source replacement for vibeops ConfigManager.

    Brise reads `CAUSE_ENTITY_WEIGHT_ENABLED` from a global ConfigManager that
    pulls from env / yaml / nacos. We collapse that to a dataclass field; users
    flip it via constructor or `os.environ.get`.
    """

    cause_entity_weight_enabled: bool = True


def compute_cause_entity_weight(
    entity_evidence: EntityScoreEvidence,
    *,
    config: ScoringConfig | None = None,
) -> float:
    """F-term penalty when E missed.

    Brise rule (composite_scorer.py:754-771):
      定因正确但定界完全未命中时，说明 Agent 可能只是"猜对了类型"但
      没有实际定位到故障实体，应对定因得分做折扣。
      系数 0.6（可通过 CAUSE_ENTITY_WEIGHT_ENABLED=false 关闭）。

    Returns 1.0 (no penalty) or 0.6 (penalty applied).
    """
    cfg = config or ScoringConfig()
    if not cfg.cause_entity_weight_enabled:
        return 1.0

    # 定界=未命中 且 非 LLM 提取失败（llm_failed 说明 LLM 抽取阶段就失败了，不惩罚）
    if (
        entity_evidence.entity_grade == "未命中"
        and entity_evidence.extraction_method != "llm_failed"
    ):
        return 0.6
    return 1.0


PROCESS_NA_SOURCES = frozenset({"rca_judge_failed", "preview_process_disabled"})


def credibility_multiplier(level: str) -> float:
    """Map conclusion reliability to the current Brise multiplier."""
    if level == "UNRELIABLE":
        return 0.2
    if level == "QUESTIONABLE":
        return 0.5
    return 1.0


def compute_weighted_result(
    *,
    fault_evidence: FaultTypeScoreEvidence,
    entity_evidence: EntityScoreEvidence,
    process_evidence: ProcessScoreEvidence,
    credibility_evidence: CredibilityScoreEvidence,
    weights: ScoreWeights | None = None,
    config: ScoringConfig | None = None,
) -> RcaScoreResult:
    """Verbatim port of brise composite_scorer._compute_weighted_result.

    The only structural change is signature: brise marks args kw-only and
    requires ScoreWeights; we default to brise production weights (0.3 / 0.4 / 0.3).
    """
    w = weights or ScoreWeights()
    w_ft = w.fault_type_weight
    w_ent = w.entity_weight
    w_proc = w.process_weight
    process_na = process_evidence.score_source in PROCESS_NA_SOURCES
    if process_na:
        w_proc = 0.0
    w_total = w_ft + w_ent + w_proc
    if w_total <= 0:
        raise ValueError(
            "score weights sum to 0 "
            f"(fault_type={w_ft}, entity={w_ent}, process={w_proc})"
        )
    w_ft_norm = w_ft / w_total
    w_ent_norm = w_ent / w_total
    w_proc_norm = w_proc / w_total

    # F-term entity penalty (Phase 4 Task 6 in brise)
    entity_weight = compute_cause_entity_weight(entity_evidence, config=config)

    raw_composite = (
        fault_evidence.score * entity_weight * w_ft_norm * 100
        + entity_evidence.score * w_ent_norm * 100
        + process_evidence.score * w_proc_norm * 100
    )
    raw_composite = round(max(0.0, min(100.0, raw_composite)), 1)

    # Credibility multiplicative gate
    gate_multiplier = credibility_multiplier(
        credibility_evidence.conclusion_reliability
    )
    overall = round(raw_composite * gate_multiplier, 1)

    normalized_weights = NormalizedWeights(
        fault_type=round(w_ft_norm, 3),
        entity=round(w_ent_norm, 3),
        process=round(w_proc_norm, 3),
    )

    fault_term_str = f"{fault_evidence.score * 100:.1f}"
    if entity_weight < 1.0:
        fault_term_str += f"×entity_w({entity_weight})"
    formula = (
        f"综合分 = {fault_term_str}×{w_ft_norm:.0%}"
        f" + {entity_evidence.score * 100:.1f}×{w_ent_norm:.0%}"
        f" + {process_evidence.score * 100:.1f}×{w_proc_norm:.0%}"
        f" = {raw_composite:.1f}"
    )
    if gate_multiplier < 1.0:
        formula += f" × 门禁(×{gate_multiplier}) = {overall:.1f}"

    return RcaScoreResult(
        scoring_version="rca",
        fault_type=fault_evidence,
        entity=entity_evidence,
        process=process_evidence,
        credibility=credibility_evidence,
        weights=normalized_weights,
        overall_score=overall,
        overall_formula=formula,
    )
