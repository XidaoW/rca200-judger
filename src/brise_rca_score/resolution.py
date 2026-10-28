"""Shared GT reasoning, fault resolution, and entity normalization helpers."""

from __future__ import annotations

from importlib.resources import files

import yaml

from .contracts import (
    ClaimAtom,
    Diagnosis,
    FaultCandidate,
    GroundTruth,
    GtReasoningStep,
    ResolvedClaimAtom,
    RootCause,
)
from .llm.base import LLMClient
from .scoring.fault_type import ReasoningStep
from .scoring.models import FaultTypeScoreEvidence, TargetEntity
from .scoring.semantic_map import SemanticFaultMapping, map_unresolved_fault_text


def _gt_reasoning_steps(gt: GroundTruth) -> list[GtReasoningStep]:
    raw_steps = gt.reasoning.get("steps")
    if not isinstance(raw_steps, list):
        return []
    return [
        GtReasoningStep.model_validate(raw)
        for raw in raw_steps
        if isinstance(raw, dict)
    ]


def _to_reasoning_step(item: GtReasoningStep) -> ReasoningStep:
    return ReasoningStep(
        step=item.step,
        target=item.target,
        fault_id=item.fault_id,
        step_type=item.step_type,
        title=item.title,
        description=item.description,
        observability=item.observability,
    )


def _reasoning_steps(gt: GroundTruth) -> list[ReasoningStep]:
    return [_to_reasoning_step(item) for item in _gt_reasoning_steps(gt)]


def _reasoning_steps_for_root(
    gt: GroundTruth,
    rc_id: str,
) -> list[ReasoningStep]:
    return [
        _to_reasoning_step(item)
        for item in _gt_reasoning_steps(gt)
        if not item.rc_refs or rc_id in item.rc_refs
    ]


def _build_fault_type_reference() -> str:
    text = (files("brise_rca_score.data") / "fault_type_catalog.yaml").read_text(
        encoding="utf-8"
    )
    catalog = yaml.safe_load(text) or {}
    lines = []
    for item in catalog.get("types") or []:
        fault_id = str(item.get("fault_id") or "").strip()
        description = str(
            item.get("description") or item.get("canonical") or ""
        ).strip()
        if fault_id:
            lines.append(f"- {fault_id}: {description}")
    return "\n".join(lines)


def _fault_dictionary() -> dict[str, str]:
    text = (files("brise_rca_score.data") / "fault_type_catalog.yaml").read_text(
        encoding="utf-8"
    )
    catalog = yaml.safe_load(text) or {}
    aliases: dict[str, str] = {}
    for item in catalog.get("types") or []:
        fault_id = str(item.get("fault_id") or "").strip()
        canonical = str(item.get("canonical") or "").strip()
        if not fault_id:
            continue
        aliases[fault_id.lower()] = fault_id
        if canonical:
            aliases[canonical.lower()] = fault_id
    return aliases


def _fault_reference(root: RootCause) -> str:
    base = _build_fault_type_reference()
    if root.fault_id.lower() in base.lower():
        return base
    description = root.conclusion or root.fault_id
    return f"{base}\n- {root.fault_id}: {description}"


def _semantic_dictionary_for_roots(roots: list[RootCause]) -> dict[str, str]:
    dictionary = dict(_fault_dictionary())
    for root in roots:
        fault_id = root.fault_id.strip()
        if not fault_id:
            continue
        dictionary[fault_id.lower()] = fault_id
        if "-" in fault_id:
            dictionary.setdefault(fault_id.split("-", 1)[0].lower(), fault_id)
    return dictionary


def _extract_actual_faults(
    diagnosis: Diagnosis,
    root: RootCause,
    *,
    llm: LLMClient,
    model: str,
    temperature: float,
    vote_count: int,
) -> tuple[list[str], list[float], list[SemanticFaultMapping]]:
    candidates = list(diagnosis.fault_candidates)
    if not candidates and diagnosis.conclusion.text.strip():
        candidates = [
            FaultCandidate(
                raw_text=diagnosis.conclusion.text,
                evidence_quote=diagnosis.conclusion.text,
                confidence=0.5,
            )
        ]
    dictionary = _semantic_dictionary_for_roots([root])
    reference = _fault_reference(root)
    faults: list[str] = []
    confidences: list[float] = []
    mappings: list[SemanticFaultMapping] = []
    for candidate in candidates:
        explicit = candidate.fault_id.strip()
        if explicit:
            faults.append(explicit)
            confidences.append(candidate.confidence)
            continue
        raw_text = (
            candidate.raw_text or candidate.evidence_quote or diagnosis.conclusion.text
        ).strip()
        mapping = map_unresolved_fault_text(
            llm=llm,
            raw_text=raw_text,
            reference=reference,
            dictionary=dictionary,
            context=diagnosis.conclusion.text[:600],
            model=model,
            temperature=temperature,
            vote_count=vote_count,
        )
        mappings.append(mapping)
        value = mapping.fault_id if mapping.status == "hit" else raw_text
        if value:
            faults.append(value)
            confidences.append(candidate.confidence)
    return faults, confidences, mappings


def _stamp_semantic_mapping(
    evidence: FaultTypeScoreEvidence,
    mappings: list[SemanticFaultMapping],
) -> None:
    if not mappings:
        return
    selected = next(
        (mapping for mapping in mappings if mapping.status == "hit"),
        next(
            (mapping for mapping in mappings if mapping.status == "llm_failed"),
            mappings[0],
        ),
    )
    evidence.semantic_mapping_status = selected.status
    evidence.semantic_mapped_fault_id = selected.fault_id
    evidence.semantic_mapping_reason = selected.reason
    evidence.semantic_mapping_votes = [
        vote.fault_id if vote is not None else "<invalid>"
        for mapping in mappings
        for vote in mapping.votes
    ]
    evidence.semantic_mapping_prompt_version = selected.prompt_version
    evidence.semantic_mapping_prompt_sha256 = selected.prompt_sha256


def _claim_atoms(diagnosis: Diagnosis) -> list[ClaimAtom]:
    if diagnosis.atoms:
        return sorted(diagnosis.atoms, key=lambda item: item.ordinal)
    entities_by_rank = {item.rank: item for item in diagnosis.entity_candidates}
    return [
        ClaimAtom(
            atom_id=f"a{candidate.rank + 1}",
            ordinal=candidate.rank,
            raw_fault_text=candidate.raw_text or candidate.evidence_quote,
            raw_entity_name=(
                entities_by_rank[candidate.rank].entity_name
                if candidate.rank in entities_by_rank
                else ""
            ),
            confidence=candidate.confidence,
            fault_id=candidate.fault_id,
        )
        for candidate in diagnosis.fault_candidates
    ]


def _fault_reference_for_roots(roots: list[RootCause]) -> str:
    reference = _build_fault_type_reference()
    reference_lower = reference.lower()
    additions = [
        f"- {root.fault_id}: {root.conclusion or root.fault_id}"
        for root in roots
        if root.fault_id.lower() not in reference_lower
    ]
    return "\n".join([reference, *additions])


def _resolve_multi_atoms(
    diagnosis: Diagnosis,
    roots: list[RootCause],
    *,
    llm: LLMClient,
    model: str,
    temperature: float,
    vote_count: int,
) -> list[ResolvedClaimAtom]:
    reference = _fault_reference_for_roots(roots)
    dictionary = _semantic_dictionary_for_roots(roots)
    resolved: list[ResolvedClaimAtom] = []
    for atom in _claim_atoms(diagnosis):
        fault_id = atom.fault_id.strip()
        mapping: SemanticFaultMapping | None = None
        if not fault_id:
            mapping = map_unresolved_fault_text(
                llm=llm,
                raw_text=atom.raw_fault_text,
                reference=reference,
                dictionary=dictionary,
                context=diagnosis.conclusion.text[:600],
                model=model,
                temperature=temperature,
                vote_count=vote_count,
            )
            fault_id = (
                mapping.fault_id
                if mapping.status == "hit"
                else atom.raw_fault_text.strip() or "UNMAPPED"
            )
        resolved.append(
            ResolvedClaimAtom(
                atom_id=atom.atom_id,
                ordinal=atom.ordinal,
                fault_id=fault_id,
                raw_fault_text=atom.raw_fault_text,
                raw_entity_name=atom.raw_entity_name,
                confidence=atom.confidence,
                semantic_mapping=mapping,
            )
        )
    return _dedupe_multi_atoms(resolved)


def _dedupe_multi_atoms(atoms: list[ResolvedClaimAtom]) -> list[ResolvedClaimAtom]:
    deduped: dict[tuple[str, str], ResolvedClaimAtom] = {}
    for atom in sorted(atoms, key=lambda item: item.ordinal):
        key = (
            atom.fault_id.strip().lower(),
            _normalized_entity_name(atom.raw_entity_name),
        )
        current = deduped.get(key)
        if current is None or atom.confidence > current.confidence:
            deduped[key] = atom
    return sorted(deduped.values(), key=lambda item: item.ordinal)


def _expected_entities(root: RootCause) -> list[TargetEntity]:
    refs = list(root.entity_group.entity_refs)
    if refs:
        return [
            TargetEntity(
                entity_id=item.entity_id,
                entity_name=item.entity_name,
                entity_type=item.entity_type,
                entity_domain=item.entity_domain,
            )
            for item in refs
        ]
    return [
        TargetEntity(entity_id=item) for item in root.entity_group.required_entity_ids
    ]


def _normalized_entity_name(value: str) -> str:
    return value.strip().rstrip("/").rsplit("/", 1)[-1].lower()


def _actual_entities(
    diagnosis: Diagnosis,
    expected: list[TargetEntity],
) -> list[TargetEntity]:
    """Resolve namespace-qualified names against GT-local entity references."""
    expected_by_name = {
        _normalized_entity_name(item.entity_name): item
        for item in expected
        if item.entity_name.strip()
    }
    resolved: list[TargetEntity] = []
    for item in diagnosis.entity_candidates:
        name = item.entity_name.strip()
        if not name:
            continue
        matched = expected_by_name.get(_normalized_entity_name(name))
        if matched is not None and "/" in name:
            resolved.append(matched.model_copy())
        else:
            resolved.append(TargetEntity(entity_name=name))
    return resolved


def _agent_chain_text(diagnosis: Diagnosis) -> str:
    return "\n".join(
        f"- {item.entity_name}: {item.detail}".strip()
        for item in diagnosis.agent_chain.nodes
        if item.entity_name or item.detail
    )


def _gt_mechanism(steps: list[ReasoningStep]) -> str:
    return "\n".join(
        f"{step.title}：{step.description}".strip("：")
        for step in steps
        if step.step_type == "cause" and step.description.strip()
    )


def _render_gt_context(steps: list[ReasoningStep]) -> tuple[str, str]:
    chain = " → ".join(
        f"{step.target or '?'}: {step.description or step.title}" for step in steps
    )
    checkpoints = "\n".join(
        f"- [P0] {step.description or step.title}" for step in steps if step.description
    )
    return chain or "无因果传播链", checkpoints or "无预期排查节点"
