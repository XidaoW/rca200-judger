"""Multi-root assignment, rescue, process scoring, and result assembly."""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    ClaimAtom,
    Conclusion,
    Diagnosis,
    EntityCandidate,
    FaultCandidate,
    GroundTruth,
    GroundTruthBundle,
    OfflineScore,
    ResolvedClaimAtom,
    RootCause,
)
from .llm.base import LLMClient
from .process_runtime import _score_process_live
from .resolution import (
    _actual_entities,
    _expected_entities,
    _fault_dictionary,
    _reasoning_steps_for_root,
    _resolve_multi_atoms,
    _stamp_semantic_mapping,
)
from .scoring.entity import score_entity
from .scoring.fault_type import score_fault_type
from .scoring.kernel import credibility_multiplier
from .scoring.models import (
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    ProcessScoreEvidence,
)
from .scoring.multi_root import (
    AtomInput,
    MultiRootResult,
    RootInput,
    compute_multi_root,
)
from .single_root import _load_topology, _resolve_entities_in_topology
from .topology.base import TopologyGraph


def _multi_atom_diagnosis(
    atom: ResolvedClaimAtom,
    case_id: str,
) -> Diagnosis:
    return Diagnosis(
        case_id=case_id,
        extraction_status="ok",
        atoms=[
            ClaimAtom(
                atom_id=atom.atom_id,
                ordinal=atom.ordinal,
                raw_fault_text=atom.raw_fault_text,
                raw_entity_name=atom.raw_entity_name,
                confidence=atom.confidence,
                fault_id=atom.fault_id,
            )
        ],
        conclusion=Conclusion(text=atom.raw_fault_text),
        fault_candidates=[
            FaultCandidate(
                fault_id=atom.fault_id,
                raw_text=atom.raw_fault_text,
                confidence=atom.confidence,
            )
        ],
        entity_candidates=[
            EntityCandidate(
                entity_name=atom.raw_entity_name,
                confidence=atom.confidence,
            )
        ],
    )


def _multi_pair_score(
    root_payload: object,
    atom_payload: object,
    *,
    gt: GroundTruth,
    topology: TopologyGraph | None,
) -> tuple[FaultTypeScoreEvidence, EntityScoreEvidence]:
    root = RootCause.model_validate(root_payload)
    atom = ResolvedClaimAtom.model_validate(atom_payload)
    fault = score_fault_type(
        expected_fault_type=root.fault_id,
        actual_fault_types=[atom.fault_id],
        actual_confidences=[atom.confidence],
        dictionary=_fault_dictionary(),
        gt_causal_chain=_reasoning_steps_for_root(gt, root.rc_id),
        llm=None,
    )
    if atom.semantic_mapping is not None:
        _stamp_semantic_mapping(fault, [atom.semantic_mapping])
    expected = _expected_entities(root)
    actual = _actual_entities(
        Diagnosis(
            entity_candidates=[EntityCandidate(entity_name=atom.raw_entity_name)]
        ),
        expected,
    )
    entity = score_entity(
        expected_entities=expected,
        extracted_entities=_resolve_entities_in_topology(actual, topology),
        topology=topology,
    )
    return fault, entity


def _multi_rescue_score(
    root_payload: object,
    atom_payload: object,
    *,
    gt: GroundTruth,
    llm: LLMClient,
) -> float:
    root = RootCause.model_validate(root_payload)
    atom = ResolvedClaimAtom.model_validate(atom_payload)
    evidence = score_fault_type(
        expected_fault_type=root.fault_id,
        actual_fault_types=[atom.fault_id],
        actual_confidences=[atom.confidence],
        dictionary=_fault_dictionary(),
        gt_causal_chain=_reasoning_steps_for_root(gt, root.rc_id),
        llm=llm,
        expected_desc=root.conclusion,
        actual_descs={atom.fault_id.lower(): atom.raw_fault_text},
        agent_conclusion=atom.raw_fault_text,
    )
    return min(0.5, max(0.0, evidence.score))


def _multi_process_score(
    root_payload: object,
    atom_payload: object,
    fault: FaultTypeScoreEvidence,
    *,
    gt: GroundTruth,
    case_id: str,
    process_diagnosis: Diagnosis,
    llm: LLMClient,
    model: str,
    temperature: float,
) -> ProcessScoreEvidence:
    root = RootCause.model_validate(root_payload)
    atom = ResolvedClaimAtom.model_validate(atom_payload)
    atom_diagnosis = _multi_atom_diagnosis(atom, case_id).model_copy(
        update={
            "analysis_output": process_diagnosis.analysis_output,
            "tool_calls_summary": process_diagnosis.tool_calls_summary,
        }
    )
    evidence = _score_process_live(
        llm=llm,
        diagnosis=atom_diagnosis,
        root=root,
        steps=_reasoning_steps_for_root(gt, root.rc_id),
        model=model,
        temperature=temperature,
    )
    if fault.score == 0 and evidence.score > 0.4:
        evidence.capped_from = evidence.score
        evidence.score = 0.4
        evidence.cap_reasons.append("定因未命中：推理链导向错误根因")
    return evidence


def _build_multi_root_result(
    *,
    result: MultiRootResult,
    diagnosis: Diagnosis,
    gt_file: Path,
    scored_output_source: str,
    digest: str,
    model: str,
) -> OfflineScore:
    process_score = result.process if result.process is not None else 0.0
    return OfflineScore(
        case_id=diagnosis.case_id or gt_file.parent.name,
        scoring_formula_version="rca-multi-local-v3.1",
        input_sha256=digest,
        judge_model=model,
        scores={
            "fault_type": result.fault,
            "entity": result.entity,
            "process": process_score,
            "overall": result.overall,
        },
        gate={"passed": result.overall >= 60.0, "threshold": 60.0},
        details={
            "fault_type": FaultTypeScoreEvidence(
                score=result.fault,
                match_tier="multi_root",
                scoring_method="multi_root_assignment",
            ).model_dump(mode="json"),
            "entity": EntityScoreEvidence(
                score=result.entity,
                entity_grade="多根因聚合",
                expected_count=result.root_count,
                matched_count=sum(
                    1 for item in result.assignment if item.atom_id is not None
                ),
            ).model_dump(mode="json"),
            "process": ProcessScoreEvidence(
                score=process_score,
                score_source=(
                    "multi_rca_review"
                    if result.process is not None
                    else "rca_judge_failed"
                ),
            ).model_dump(mode="json"),
            "multi_root": result.model_dump(mode="json"),
            "formula": "multi-root weighted assignment: F×0.3 + E×0.4 + P×0.3",
        },
        source={
            "gt": str(gt_file),
            "scored_output": scored_output_source,
            "scoring": "dashscope_multi_root",
        },
    )


def _score_multi_root_case(
    *,
    bundle: GroundTruthBundle,
    diagnosis: Diagnosis,
    gt_file: Path,
    scored_output_source: str,
    digest: str,
    llm: LLMClient,
    model: str,
    temperature: float,
    credibility: str,
    topology_file: Path | None,
    replay_process_file: Path | None,
    vote_count: int,
) -> OfflineScore:
    if replay_process_file is not None:
        raise NotImplementedError(
            "multi-root process replay requires per-root review evidence"
        )
    roots = bundle.ground_truth.outcome.root_causes
    atoms = _resolve_multi_atoms(
        diagnosis,
        roots,
        llm=llm,
        model=model,
        temperature=temperature,
        vote_count=vote_count,
    )
    topology = _load_topology(topology_file)
    result = compute_multi_root(
        roots=[RootInput(rc_id=root.rc_id, payload=root) for root in roots],
        atoms=[
            AtomInput(atom_id=atom.atom_id, ordinal=atom.ordinal, payload=atom)
            for atom in atoms
        ],
        pair_scorer=lambda root, atom: _multi_pair_score(
            root,
            atom,
            gt=bundle.ground_truth,
            topology=topology,
        ),
        rescue_scorer=lambda root, atom: _multi_rescue_score(
            root,
            atom,
            gt=bundle.ground_truth,
            llm=llm,
        ),
        process_scorer=lambda root, atom, fault: _multi_process_score(
            root,
            atom,
            fault,
            gt=bundle.ground_truth,
            case_id=diagnosis.case_id,
            process_diagnosis=diagnosis,
            llm=llm,
            model=model,
            temperature=temperature,
        ),
        credibility_multiplier=credibility_multiplier(credibility.upper()),
    )
    return _build_multi_root_result(
        result=result,
        diagnosis=diagnosis,
        gt_file=gt_file,
        scored_output_source=scored_output_source,
        digest=digest,
        model=model,
    )
