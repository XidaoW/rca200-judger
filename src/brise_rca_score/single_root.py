"""Single-root dimension scoring and result assembly."""

from __future__ import annotations

from pathlib import Path

from .contracts import Diagnosis, OfflineScore, RootCause
from .llm.base import LLMClient
from .resolution import (
    _actual_entities,
    _agent_chain_text,
    _expected_entities,
    _extract_actual_faults,
    _fault_dictionary,
    _gt_mechanism,
    _stamp_semantic_mapping,
)
from .scoring.entity import score_entity
from .scoring.fault_type import ReasoningStep, score_fault_type
from .scoring.kernel import compute_weighted_result
from .scoring.mechanism import apply_mechanism_outcome, evaluate_mechanism
from .scoring.models import (
    CredibilityScoreEvidence,
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    ProcessScoreEvidence,
    TargetEntity,
)
from .topology import JsonTopology, TopologyGraph


def _score_fault_dimension(
    *,
    diagnosis: Diagnosis,
    root: RootCause,
    steps: list[ReasoningStep],
    llm: LLMClient,
    model: str,
    temperature: float,
    vote_count: int,
) -> FaultTypeScoreEvidence:
    actual_faults, confidences, mappings = _extract_actual_faults(
        diagnosis,
        root,
        llm=llm,
        model=model,
        temperature=temperature,
        vote_count=vote_count,
    )
    evidence = score_fault_type(
        expected_fault_type=root.fault_id,
        actual_fault_types=actual_faults,
        actual_confidences=confidences,
        dictionary=_fault_dictionary(),
        gt_causal_chain=steps,
        llm=llm,
        expected_desc=root.conclusion,
        agent_chain_text=_agent_chain_text(diagnosis),
        agent_conclusion=diagnosis.conclusion.text,
    )
    _stamp_semantic_mapping(evidence, mappings)
    if not actual_faults or evidence.score <= 0:
        evidence.mechanism_gate_status = "skipped_low_score"
        return evidence
    if evidence.causal_match_kind or evidence.match_tier.startswith(
        ("causal_", "merged_", "llm_")
    ):
        evidence.mechanism_gate_status = "skipped_hop_tier"
        return evidence
    outcome = evaluate_mechanism(
        llm=llm,
        expected=root.fault_id,
        expected_desc=root.conclusion,
        gt_mechanism=_gt_mechanism(steps),
        actual_list=actual_faults,
        agent_conclusion=diagnosis.conclusion.text,
        agent_chain_text=_agent_chain_text(diagnosis),
        model=model,
        temperature=temperature,
        vote_count=vote_count,
    )
    return apply_mechanism_outcome(evidence, outcome)


def _load_topology(topology_file: Path | None) -> TopologyGraph | None:
    if topology_file is None:
        return None

    return JsonTopology.from_file(topology_file)


def _resolve_entities_in_topology(
    entities: list[TargetEntity], topology: TopologyGraph | None
) -> list[TargetEntity]:
    if topology is None:
        return entities
    resolved: list[TargetEntity] = []
    for entity in entities:
        if entity.entity_id:
            resolved.append(entity)
            continue
        cid = topology.find_node_by_name(entity.entity_name)
        node = topology.get_node(cid) if cid else None
        if node is None:
            resolved.append(entity)
            continue
        resolved.append(
            TargetEntity(
                entity_id=node.entity_id or node.composite_id,
                entity_name=node.entity_name,
                entity_domain=node.entity_domain,
                entity_type=node.entity_type,
            )
        )
    return resolved


def _score_entity_dimension(
    root: RootCause,
    diagnosis: Diagnosis,
    topology_file: Path | None,
) -> EntityScoreEvidence:
    topology = _load_topology(topology_file)
    expected = _expected_entities(root)
    return score_entity(
        expected_entities=expected,
        extracted_entities=_resolve_entities_in_topology(
            _actual_entities(diagnosis, expected), topology
        ),
        topology=topology,
    )


def _build_offline_result(
    *,
    diagnosis: Diagnosis,
    gt_file: Path,
    scored_output_source: str,
    scoring_source: str,
    digest: str,
    model: str,
    credibility: str,
    fault: FaultTypeScoreEvidence,
    entity: EntityScoreEvidence,
    process: ProcessScoreEvidence,
) -> OfflineScore:
    result = compute_weighted_result(
        fault_evidence=fault,
        entity_evidence=entity,
        process_evidence=process,
        credibility_evidence=CredibilityScoreEvidence(
            conclusion_reliability=credibility.upper()
        ),
    )
    return OfflineScore(
        case_id=diagnosis.case_id or gt_file.parent.name,
        input_sha256=digest,
        judge_model=model,
        scores={
            "fault_type": fault.score,
            "entity": entity.score,
            "process": process.score,
            "overall": result.overall_score,
        },
        gate={"passed": result.overall_score >= 60.0, "threshold": 60.0},
        details={
            "fault_type": fault.model_dump(mode="json"),
            "entity": entity.model_dump(mode="json"),
            "process": process.model_dump(mode="json"),
            "formula": result.overall_formula,
        },
        source={
            "gt": str(gt_file),
            "scored_output": scored_output_source,
            "scoring": scoring_source,
        },
    )
