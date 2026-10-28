"""rcaeval.scoring.entity — E (entity) scorer with 7-grade tiered matching.

Ported from cms2.0-route-agent/vibeops/eval/rca_evaluate/engine/scoring/
entity_scorer.py + topology_distance.py.

Pipeline (matches brise score_entity, lines 38-158):

    1. String fallback (always runs):
       a. EXACT (100): entity_name lower-equals
       b. SERVICE (75): K8s pod name `<svc>-<10hex>-<5alnum>` → `<svc>`,
          tried both directions (agent=pod/gt=svc and vice versa)

    2. Topology distance (when TopologyGraph supplied):
       - Resolve each entity to a graph node (find_node_by_name)
       - BFS distance → hop_distance_to_grade:
           hop=0 → SERVICE (75)        # same service node
           hop=1 → NEIGHBOR_1 (40)
           hop=2 → NEIGHBOR_2 (30)
           hop=3 → NEIGHBOR_3 (20)
           None  → fallback: same_service tie-breaker → SERVICE (75)
                   or NONE (0)

    3. Take max(string_fallback, topology) for each (gt × agent) pair, then
       max across all pairs. RCA convention: finding ANY correct entity suffices.

Note on HOST_NODE (60): present in EntityGrade enum + docstring but never
awarded by brise's actual scoring path. Left in enum for forward-compat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..topology.base import TopologyGraph
from .models import EntityGrade, EntityScoreEvidence, TargetEntity

_K8S_POD_RE = re.compile(r"^(.+)-[a-f0-9]{10}-[a-z0-9]{5}$")
_K8S_RS_RE = re.compile(r"^(.+)-[a-f0-9]{10}$")


def _k8s_service_prefix(name: str) -> str | None:
    """`inventory-5c4b7bcb9c-d24qh` → `inventory`; otherwise None."""
    m = _K8S_POD_RE.match(name)
    if m:
        return m.group(1)
    m = _K8S_RS_RE.match(name)
    if m:
        return m.group(1)
    return None


def _hop_distance_to_grade(hop: int | None) -> tuple[float, EntityGrade]:
    """Brise topology_distance.py:hop_distance_to_grade — verbatim mapping."""
    if hop is None:
        return 0.0, EntityGrade.NONE
    if hop == 0:
        return 75.0, EntityGrade.SERVICE
    if hop == 1:
        return 40.0, EntityGrade.NEIGHBOR_1
    if hop == 2:
        return 30.0, EntityGrade.NEIGHBOR_2
    if hop <= 3:
        return 20.0, EntityGrade.NEIGHBOR_3
    return 0.0, EntityGrade.NONE


@dataclass
class _MatchResult:
    score_100: float
    grade: EntityGrade
    gt_id: str = ""
    agent_name: str = ""
    method: str = ""


def _string_fallback_match(
    expected: list[TargetEntity],
    extracted: list[TargetEntity],
) -> _MatchResult:
    """Brise _string_fallback_match (lines 185-236): EXACT then K8s pod→svc prefix."""
    # 1. EXACT
    for gt in expected:
        gt_name = (gt.entity_name or gt.entity_id or "").strip().lower()
        for agent in extracted:
            agent_name = (agent.entity_name or agent.entity_id or "").strip().lower()
            if gt_name and agent_name and gt_name == agent_name:
                return _MatchResult(
                    score_100=100.0,
                    grade=EntityGrade.EXACT,
                    gt_id=gt.entity_id,
                    agent_name=agent.entity_name or agent.entity_id,
                    method="string_exact",
                )

    # 2. K8s pod prefix → SERVICE (both directions)
    for gt in expected:
        gt_name = (gt.entity_name or gt.entity_id or "").strip().lower()
        for agent in extracted:
            agent_name = (agent.entity_name or agent.entity_id or "").strip().lower()
            if not gt_name or not agent_name:
                continue
            agent_prefix = _k8s_service_prefix(agent_name)
            if agent_prefix and agent_prefix == gt_name:
                return _MatchResult(
                    score_100=75.0,
                    grade=EntityGrade.SERVICE,
                    gt_id=gt.entity_id,
                    agent_name=agent.entity_name or agent.entity_id,
                    method="k8s_pod_prefix",
                )
            gt_prefix = _k8s_service_prefix(gt_name)
            if gt_prefix and gt_prefix == agent_name:
                return _MatchResult(
                    score_100=75.0,
                    grade=EntityGrade.SERVICE,
                    gt_id=gt.entity_id,
                    agent_name=agent.entity_name or agent.entity_id,
                    method="k8s_pod_prefix",
                )

    return _MatchResult(score_100=0.0, grade=EntityGrade.NONE)


def _topology_distance_match(
    topology: TopologyGraph,
    expected: list[TargetEntity],
    extracted: list[TargetEntity],
) -> _MatchResult:
    """Per-pair topology distance, take max across (gt × agent).

    Grade precedence when multiple checks fire:
      EXACT (id match) > SERVICE (hop=0 OR same_service) > HOST_NODE (same_host_node)
       > NEIGHBOR_1/2/3 (BFS hops) > NONE.
    """
    best = _MatchResult(score_100=0.0, grade=EntityGrade.NONE)
    for gt in expected:
        gt_cid = topology.find_node_by_name(gt.entity_name or gt.entity_id)
        if not gt_cid:
            continue
        for agent in extracted:
            agent_cid = topology.find_node_by_name(agent.entity_name or agent.entity_id)
            if not agent_cid:
                continue
            hop = topology.structural_distance(agent_cid, gt_cid)
            score, grade = _hop_distance_to_grade(hop)
            if hop is None and topology.same_service(agent_cid, gt_cid):
                # Brise tie-breaker: nodes in graph but disconnected, same service
                score, grade = 75.0, EntityGrade.SERVICE
            # HOST_NODE check — fills the brise enum slot that brise never used,
            # when the local topology has explicit `hosts` edges.
            # Only consult when the BFS / SERVICE checks didn't already produce
            # something stronger (HOST_NODE = 60, weaker than SERVICE = 75).
            if score < 60.0 and hasattr(topology, "same_host_node"):
                if topology.same_host_node(agent_cid, gt_cid):  # type: ignore[attr-defined]
                    score, grade = 60.0, EntityGrade.HOST_NODE
            if score > best.score_100:
                best = _MatchResult(
                    score_100=score,
                    grade=grade,
                    gt_id=gt.entity_id,
                    agent_name=agent.entity_name or agent.entity_id,
                    method="topology",
                )
    return best


def score_entity(
    *,
    expected_entities: list[TargetEntity],
    extracted_entities: list[TargetEntity],
    topology: TopologyGraph | None = None,
    extraction_method: str = "",
) -> EntityScoreEvidence:
    """E scorer. Mirrors brise score_entity but inputs are already-extracted entities.

    Brise's score_entity has an `actual_text` parameter that triggers a full LLM
    extraction. M3 moves that out: caller is responsible for extraction
    (rcaeval.extraction.entity in M4). If you pass empty extracted_entities,
    score is 0.

    `extraction_method` is forwarded into EntityScoreEvidence to drive the
    kernel's entity penalty rule (set to "llm_failed" to disable the 0.6
    multiplier on F when E missed).
    """
    if not expected_entities:
        return EntityScoreEvidence(
            score=0.0,
            entity_grade="未命中",
            entity_grade_score=0.0,
            expected_count=0,
            matched_count=0,
            extraction_method=extraction_method,
            reason="缺少 GT 实体，跳过定界评分。",
        )

    if not extracted_entities:
        return EntityScoreEvidence(
            score=0.0,
            entity_grade="未命中",
            entity_grade_score=0.0,
            expected_count=len(expected_entities),
            matched_count=0,
            extraction_method=extraction_method,
            reason="Agent 未提取出任何实体。",
        )

    # Brise only allows exact-name fallback when topology is unavailable.
    # With topology present, canonical IDs / structure decide the grade; equal
    # names on the same graph node are service-level (hop=0), not exact.
    if topology is None:
        best = _string_fallback_match(expected_entities, extracted_entities)
    else:
        best = _MatchResult(score_100=0.0, grade=EntityGrade.NONE)
        for gt in expected_entities:
            for agent in extracted_entities:
                if gt.entity_id and agent.entity_id and gt.entity_id == agent.entity_id:
                    best = _MatchResult(
                        score_100=100.0,
                        grade=EntityGrade.EXACT,
                        gt_id=gt.entity_id,
                        agent_name=agent.entity_name or agent.entity_id,
                        method="canonical_id",
                    )
                    break
            if best.score_100 == 100.0:
                break
        prefix_match = _string_fallback_match(expected_entities, extracted_entities)
        if (
            prefix_match.grade is EntityGrade.SERVICE
            and prefix_match.score_100 > best.score_100
        ):
            best = prefix_match
        topo_match = _topology_distance_match(
            topology, expected_entities, extracted_entities
        )
        if topo_match.score_100 > best.score_100:
            best = topo_match

    return EntityScoreEvidence(
        score=best.score_100 / 100.0,
        entity_grade=best.grade.label,
        entity_grade_score=best.score_100,
        expected_count=len(expected_entities),
        matched_count=1 if best.gt_id else 0,
        matched_entities=[best.gt_id] if best.gt_id else [],
        entity_resolve_method=best.method,
        extraction_method=extraction_method,
        reason=(
            f"定界 {int(best.score_100)}/100（{best.grade.label}）："
            f"命中 {1 if best.gt_id else 0}/{len(expected_entities)}"
        ),
    )
