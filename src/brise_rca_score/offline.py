"""Backward-compatible facade for the standalone local scorer.

Implementation is split by responsibility; existing imports from
``brise_rca_score.offline`` remain supported.
"""

from .api import score_case
from .contracts import (
    AgentCausalLink,
    AgentChain,
    AgentOutput,
    AgentRootCause,
    ChainNode,
    ClaimAtom,
    Conclusion,
    Diagnosis,
    EntityCandidate,
    EntityGroup,
    EntityRef,
    FaultCandidate,
    GroundTruth,
    GroundTruthBundle,
    GtReasoningStep,
    OfflineScore,
    Outcome,
    ResolvedClaimAtom,
    RootCause,
)
from .inputs import diagnosis_from_agent_output, load_case_inputs

__all__ = [
    "AgentCausalLink",
    "AgentChain",
    "AgentOutput",
    "AgentRootCause",
    "ChainNode",
    "ClaimAtom",
    "Conclusion",
    "Diagnosis",
    "EntityCandidate",
    "EntityGroup",
    "EntityRef",
    "FaultCandidate",
    "GroundTruth",
    "GroundTruthBundle",
    "GtReasoningStep",
    "OfflineScore",
    "Outcome",
    "ResolvedClaimAtom",
    "RootCause",
    "diagnosis_from_agent_output",
    "load_case_inputs",
    "score_case",
]
