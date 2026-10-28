"""rcaeval.scoring.models — M0 subset of brise's scoring data models.

Ported from cms2.0-route-agent/vibeops/eval/rca_evaluate/engine/scoring/models.py.
Verbatim copy of the enums + score tables + dataclasses used by M0:
  - FaultTypeTier (7 levels with score table)
  - EntityGrade   (7 levels with score table)
  - FaultTypeScoreEvidence / EntityScoreEvidence / ProcessScoreEvidence /
    CredibilityScoreEvidence
  - NormalizedWeights / RcaScoreResult

Stripped for M0:
  - vibeops.report.*       — not needed; we only emit JSON
  - benchmark_pool.*       — not needed; inline GT only
  - score_stability.policy / repository — ScoreWeights inlined as plain dataclass
  - metric_query_tool      — not needed at scoring layer
  - structured_agent_diagnosis_models — minimal stubs inlined below
  - TargetEntity (from engine.models) — inlined below
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inlined: brise engine.models.TargetEntity (just the fields scoring touches)
# ---------------------------------------------------------------------------
class TargetEntity(BaseModel):
    entity_id: str = ""
    entity_name: str = ""
    entity_type: str = ""
    entity_domain: str = ""


# ---------------------------------------------------------------------------
# F (fault_type) — 7-level tier table
# brise: scoring/models.py:53-87  (verbatim)
# ---------------------------------------------------------------------------
class FaultTypeTier(str, Enum):
    """Fault-type match tier."""

    EXACT = "exact"
    CLUSTER = "cluster"
    CAUSAL_1HOP = "causal_1hop"
    CAUSAL_2HOP = "causal_2hop"
    CAUSAL_3HOP = "causal_3hop"
    INJECTION_POINT = "injection_point"
    NONE = "none"

    @property
    def label(self) -> str:
        return _FAULT_TYPE_TIER_LABELS[self]

    @property
    def default_score(self) -> float:
        return _FAULT_TYPE_TIER_SCORES[self]


_FAULT_TYPE_TIER_LABELS: dict[FaultTypeTier, str] = {
    FaultTypeTier.EXACT: "精确",
    FaultTypeTier.CLUSTER: "同簇",
    FaultTypeTier.CAUSAL_1HOP: "因果链-1跳",
    FaultTypeTier.CAUSAL_2HOP: "因果链-2跳",
    FaultTypeTier.CAUSAL_3HOP: "因果链-3跳",
    FaultTypeTier.INJECTION_POINT: "注入点",
    FaultTypeTier.NONE: "未命中",
}

_FAULT_TYPE_TIER_SCORES: dict[FaultTypeTier, float] = {
    FaultTypeTier.EXACT: 1.0,
    FaultTypeTier.CLUSTER: 0.5,
    FaultTypeTier.CAUSAL_1HOP: 0.85,
    FaultTypeTier.CAUSAL_2HOP: 0.65,
    FaultTypeTier.CAUSAL_3HOP: 0.50,
    FaultTypeTier.INJECTION_POINT: 0.30,
    FaultTypeTier.NONE: 0.0,
}


# ---------------------------------------------------------------------------
# E (entity) — 7-level grade table
# brise: scoring/models.py:90-117 (verbatim)
# Score table (per docstring comment): 100 / 75 / 60 / 40 / 30 / 20 / 0
# ---------------------------------------------------------------------------
class EntityGrade(str, Enum):
    EXACT = "exact"
    SERVICE = "service_match"
    HOST_NODE = "host_node_match"
    NEIGHBOR_1 = "neighbor_1hop"
    NEIGHBOR_2 = "neighbor_2hop"
    NEIGHBOR_3 = "neighbor_3hop"
    NONE = "none"

    @property
    def label(self) -> str:
        return _ENTITY_GRADE_LABELS[self]

    @property
    def default_score_100(self) -> int:
        return _ENTITY_GRADE_SCORES_100[self]

    @property
    def default_score(self) -> float:
        return _ENTITY_GRADE_SCORES_100[self] / 100.0


_ENTITY_GRADE_LABELS: dict[EntityGrade, str] = {
    EntityGrade.EXACT: "精确命中",
    EntityGrade.SERVICE: "服务级命中",
    EntityGrade.HOST_NODE: "宿主节点命中",
    EntityGrade.NEIGHBOR_1: "近邻1跳",
    EntityGrade.NEIGHBOR_2: "近邻2跳",
    EntityGrade.NEIGHBOR_3: "近邻3跳",
    EntityGrade.NONE: "未命中",
}

_ENTITY_GRADE_SCORES_100: dict[EntityGrade, int] = {
    EntityGrade.EXACT: 100,
    EntityGrade.SERVICE: 75,
    EntityGrade.HOST_NODE: 60,
    EntityGrade.NEIGHBOR_1: 40,
    EntityGrade.NEIGHBOR_2: 30,
    EntityGrade.NEIGHBOR_3: 20,
    EntityGrade.NONE: 0,
}


# ---------------------------------------------------------------------------
# Score evidence — pydantic models matching brise's serialized output 1:1
# Field names match the brise evaluate response JSON exactly so fixture tests
# can deserialize brise output directly into these models.
# ---------------------------------------------------------------------------
class FaultTypeScoreEvidence(BaseModel):
    score: float = 0.0
    match_tier: str = ""  # FaultTypeTier value, "" if no run
    matched_fault_id: str = ""
    predicted_fault_ids: list[str] = Field(default_factory=list)
    shared_category: str = ""
    scoring_method: str = (
        ""  # "exact" / "cluster" / "causal_*hop" / "llm_rescue" / "unmatched"
    )
    reason: str = ""
    match_detail: str = ""
    causal_match_depth: int | None = None
    causal_match_kind: str = ""
    semantic_mapping_status: str = ""
    semantic_mapped_fault_id: str = ""
    semantic_mapping_reason: str = ""
    semantic_mapping_votes: list[str] = Field(default_factory=list)
    semantic_mapping_prompt_version: str = ""
    semantic_mapping_prompt_sha256: str = ""
    mechanism_gate_status: str = ""
    mechanism_gate_reason: str = ""
    mechanism_gate_claim: str = ""
    mechanism_gate_prior_score: float | None = None
    mechanism_gate_prior_tier: str = ""
    mechanism_gate_votes: list[str] = Field(default_factory=list)

    # extras from brise output that we tolerate but don't compute in M0
    expected_raw: str = ""
    expected_normalized: str = ""
    actual_raw: list[str] = Field(default_factory=list)


class EntityScoreEvidence(BaseModel):
    score: float = 0.0
    entity_grade: str = ""  # "精确命中" / "服务级命中" / ... / "未命中"
    entity_grade_score: float = 0.0
    expected_count: int = 0
    matched_count: int = 0
    matched_entities: list[str] = Field(default_factory=list)
    entity_resolve_method: str = ""
    reason: str = ""

    # extraction_method is used by O kernel's entity_penalty rule
    extraction_method: str = ""  # "llm_failed" disables the 0.6 penalty


class ProcessScoreEvidence(BaseModel):
    score: float = 0.0
    score_source: str = ""  # "rca_review" / "rca_process_judge" / "skipped"
    tier: str = ""
    tier_reason: str = ""
    review_dimensions: list[dict] = Field(default_factory=list)
    explanation_for_human: str = ""
    fault_type_used: str = ""
    reason: str = ""
    score_formula: str = ""
    samples: int = 1
    judge_model: str = ""
    capped_from: float | None = None
    cap_reasons: list[str] = Field(default_factory=list)


class CredibilityScoreEvidence(BaseModel):
    conclusion_reliability: str = (
        "RELIABLE"  # UNRELIABLE / QUESTIONABLE / RELIABLE / UNKNOWN
    )
    reasons: list[str] = Field(default_factory=list)
    cheating_score: float | None = None  # brise parity
    signals_summary: str = ""  # brise parity
    score: float = 1.0  # 0/0.5/1.0 — display only; gate uses level


# ---------------------------------------------------------------------------
# Weights / final result
# ---------------------------------------------------------------------------
class ScoreWeights(BaseModel):
    """Inlined from brise score_stability.models.ScoreWeights.

    Default tuned to match brise production: (0.3 / 0.4 / 0.3).
    """

    fault_type_weight: float = 0.3
    entity_weight: float = 0.4
    process_weight: float = 0.3


class NormalizedWeights(BaseModel):
    fault_type: float = 0.0
    entity: float = 0.0
    process: float = 0.0


class RcaScoreResult(BaseModel):
    scoring_version: str = "rca"
    fault_type: FaultTypeScoreEvidence
    entity: EntityScoreEvidence
    process: ProcessScoreEvidence
    credibility: CredibilityScoreEvidence
    weights: NormalizedWeights
    overall_score: float = 0.0
    overall_formula: str = ""
