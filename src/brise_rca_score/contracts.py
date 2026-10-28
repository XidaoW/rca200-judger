"""Pydantic contracts shared by the scorer orchestration layers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .scoring.semantic_map import SemanticFaultMapping


class EntityRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_id: str = ""
    entity_name: str = ""
    entity_domain: str = ""
    entity_type: str = ""


class EntityGroup(BaseModel):
    model_config = ConfigDict(extra="allow")

    required_entity_ids: list[str] = Field(default_factory=list)
    alias_entity_ids: list[str] = Field(default_factory=list)
    entity_refs: list[EntityRef] = Field(default_factory=list)


class RootCause(BaseModel):
    model_config = ConfigDict(extra="allow")

    rc_id: str = "rc1"
    fault_id: str
    entity_group: EntityGroup = Field(default_factory=EntityGroup)
    conclusion: str = ""


class Outcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    root_causes: list[RootCause]
    expected_conclusion: str = ""


class GtReasoningStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int = 0
    target: str = ""
    fault_id: str = ""
    step_type: str = ""
    title: str = ""
    description: str = ""
    observability: list[dict[str, object]] = Field(default_factory=list)
    rc_refs: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="allow")

    outcome: Outcome
    reasoning: dict[str, object] = Field(default_factory=dict)


class GroundTruthBundle(BaseModel):
    model_config = ConfigDict(extra="allow")

    ground_truth: GroundTruth
    workspace: str = ""
    region_id: str = ""
    anchor_time: str = ""


class Conclusion(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str = ""


class FaultCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    fault_id: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    evidence_quote: str = ""
    rank: int = 0


class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str = ""
    confidence: float = 0.0
    evidence_quote: str = ""
    rank: int = 0


class ChainNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str = ""
    detail: str = ""
    fault_id: str = ""
    role: str = "intermediate"


class AgentChain(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: list[ChainNode] = Field(default_factory=list)
    edges: list[dict[str, object]] = Field(default_factory=list)


class ClaimAtom(BaseModel):
    """One Agent fault × entity claim kept independent for multi-root scoring."""

    atom_id: str
    ordinal: int = Field(ge=0)
    raw_fault_text: str
    raw_entity_name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fault_id: str = ""


class ResolvedClaimAtom(BaseModel):
    atom_id: str
    ordinal: int
    fault_id: str
    raw_fault_text: str
    raw_entity_name: str
    confidence: float
    semantic_mapping: SemanticFaultMapping | None = None


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = ""
    extraction_status: str = "ok"
    atoms: list[ClaimAtom] = Field(default_factory=list)
    conclusion: Conclusion = Field(default_factory=Conclusion)
    fault_candidates: list[FaultCandidate] = Field(default_factory=list)
    entity_candidates: list[EntityCandidate] = Field(default_factory=list)
    agent_chain: AgentChain = Field(default_factory=AgentChain)
    analysis_output: str = ""
    tool_calls_summary: str = ""


class AgentRootCause(BaseModel):
    model_config = ConfigDict(extra="allow")

    fault_id: str | None = None
    fault_type: str | None = None
    entity_name: str | None = None
    entity_id: str | None = None
    reasoning: str | None = None
    confidence: str | None = None


class AgentCausalLink(BaseModel):
    model_config = ConfigDict(extra="allow")

    cause: str | None = None
    effect: str | None = None


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = ""
    root_causes: list[AgentRootCause] = Field(default_factory=list)
    causal_chain: list[AgentCausalLink] = Field(default_factory=list)


class OfflineScore(BaseModel):
    case_id: str
    scoring_formula_version: str = "rca-local-v3.1"
    input_sha256: str
    judge_model: str
    scores: dict[str, float]
    gate: dict[str, object]
    details: dict[str, object]
    source: dict[str, str]
