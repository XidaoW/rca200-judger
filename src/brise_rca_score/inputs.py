"""Local Ground Truth and Agent-output loading."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import (
    AgentChain,
    AgentOutput,
    ChainNode,
    ClaimAtom,
    Conclusion,
    Diagnosis,
    EntityCandidate,
    FaultCandidate,
    GroundTruthBundle,
)


def diagnosis_from_agent_output(agent_output_file: Path) -> Diagnosis:
    """Convert the RCA200 final answer to the score-ready local diagnosis contract."""
    agent = AgentOutput.model_validate_json(agent_output_file.read_bytes())
    confidence_values = {"high": 0.9, "medium": 0.6, "low": 0.3}
    fault_candidates: list[FaultCandidate] = []
    entity_candidates: list[EntityCandidate] = []
    atoms: list[ClaimAtom] = []
    nodes: list[ChainNode] = []
    edges: list[dict[str, object]] = []
    conclusions: list[str] = []
    for rank, root in enumerate(agent.root_causes):
        reasoning = str(root.reasoning or "")
        confidence = confidence_values.get(str(root.confidence or "").lower(), 0.5)
        conclusions.append(reasoning)
        fault_candidates.append(
            FaultCandidate(
                fault_id=str(root.fault_id or ""),
                raw_text=str(root.fault_type or "") or reasoning[:500],
                confidence=confidence,
                evidence_quote=reasoning[:2000],
                rank=rank,
            )
        )
        entity_name = str(root.entity_name or root.entity_id or "")
        entity_candidates.append(
            EntityCandidate(
                entity_name=entity_name,
                confidence=confidence,
                evidence_quote=reasoning[:2000],
                rank=rank,
            )
        )
        atoms.append(
            ClaimAtom(
                atom_id=f"a{rank + 1}",
                ordinal=rank,
                raw_fault_text=str(root.fault_type or "") or reasoning,
                raw_entity_name=entity_name,
                confidence=confidence,
                fault_id=str(root.fault_id or ""),
            )
        )
    for item in agent.causal_chain:
        cause = str(item.cause or "")
        effect = str(item.effect or "")
        nodes.append(
            ChainNode(
                entity_name=cause,
                detail=effect,
                role="intermediate",
            )
        )
        if len(nodes) > 1:
            edges.append(
                {
                    "from": len(nodes) - 2,
                    "to": len(nodes) - 1,
                    "relation": "causes",
                    "detail": effect,
                }
            )
    return Diagnosis(
        case_id=agent.case_id,
        extraction_status="ok" if fault_candidates else "empty",
        atoms=atoms[:10],
        conclusion=Conclusion(text="\n".join(conclusions)),
        fault_candidates=fault_candidates[:10],
        entity_candidates=entity_candidates[:10],
        agent_chain=AgentChain(nodes=nodes, edges=edges),
    )


def load_case_inputs(
    gt_file: Path,
    diagnosis_file: Path | None,
    agent_output_file: Path | None,
) -> tuple[GroundTruthBundle, Diagnosis, str, str]:
    """Load GT and either a prepared diagnosis or the original Agent output."""
    if (diagnosis_file is None) == (agent_output_file is None):
        raise ValueError("provide exactly one of diagnosis_file or agent_output_file")
    gt_bytes = gt_file.read_bytes()
    if diagnosis_file is not None:
        input_bytes = diagnosis_file.read_bytes()
        diagnosis = Diagnosis.model_validate_json(input_bytes)
        source_path = str(diagnosis_file)
    else:
        assert agent_output_file is not None
        input_bytes = agent_output_file.read_bytes()
        diagnosis = diagnosis_from_agent_output(agent_output_file)
        source_path = str(agent_output_file)
    digest = hashlib.sha256(gt_bytes + b"\0" + input_bytes).hexdigest()
    gt = GroundTruthBundle.model_validate_json(gt_bytes)
    return gt, diagnosis, digest, source_path
