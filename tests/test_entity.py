"""M3 fixture + synthetic tests: rcaeval.entity must reproduce brise E pipeline.

Coverage by grade:
  EXACT       — live fixtures e_slowSQL + e_hotspot (E=1.0) + synthetic
  SERVICE     — synthetic K8s pod prefix → svc + topology hop=0
  NEIGHBOR_1  — synthetic 1-hop calls graph
  NEIGHBOR_2  — synthetic 2-hop
  NEIGHBOR_3  — synthetic 3-hop
  NONE        — disconnected
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brise_rca_score.scoring.entity import score_entity
from brise_rca_score.scoring.models import TargetEntity
from brise_rca_score.topology.base import StubTopology

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())["data"]


# ---------------------------------------------------------------------------
# EXACT — locked against brise live fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fixture,expected_e",
    [
        ("e_slowSQL.json", 1.0),  # agent says "inventory", GT names ["inventory"]
        ("e_hotspot.json", 1.0),  # agent says "inventory", GT names ["inventory"]
    ],
)
def test_entity_exact_from_fixtures(fixture, expected_e):
    data = _load(fixture)
    gt_names = data["ground_truth"]["target_entity_names"]
    gt_ids = data["ground_truth"]["target_entity_ids"]
    gt_entities = [
        TargetEntity(entity_id=eid, entity_name=name)
        for eid, name in zip(gt_ids, gt_names)
    ]
    agent_cands = data["diagnosis"]["entity_candidates"]
    extracted = [
        TargetEntity(
            entity_id=c["resolved_entity"]["entity_id"],
            entity_name=c["resolved_entity"]["entity_name"],
            entity_type=c["resolved_entity"]["entity_type"],
            entity_domain=c["resolved_entity"]["entity_domain"],
        )
        for c in agent_cands
    ]
    brise_e = data["scores"]["entity"]
    assert brise_e == expected_e

    ev = score_entity(expected_entities=gt_entities, extracted_entities=extracted)
    assert ev.score == expected_e
    assert ev.entity_grade == "精确命中"
    assert ev.matched_count == 1


def test_entity_none_from_mq_fixture():
    """mq case: agent says 'checkout' but GT is 'fraud-detection' → E=0."""
    data = _load("e_mq.json")
    gt_entities = [
        TargetEntity(entity_id=eid, entity_name=name)
        for eid, name in zip(
            data["ground_truth"]["target_entity_ids"],
            data["ground_truth"]["target_entity_names"],
        )
    ]
    extracted = [
        TargetEntity(
            entity_id=c["resolved_entity"]["entity_id"],
            entity_name=c["resolved_entity"]["entity_name"],
        )
        for c in data["diagnosis"]["entity_candidates"]
    ]
    assert data["scores"]["entity"] == 0.0

    ev = score_entity(expected_entities=gt_entities, extracted_entities=extracted)
    assert ev.score == 0.0
    assert ev.entity_grade == "未命中"


# ---------------------------------------------------------------------------
# SERVICE — K8s pod prefix
# ---------------------------------------------------------------------------
def test_entity_service_k8s_pod_prefix():
    """Agent says full pod name, GT is service → SERVICE (75)."""
    gt = [TargetEntity(entity_id="svc:inventory", entity_name="inventory")]
    extracted = [
        TargetEntity(
            entity_id="pod:inventory-5c4b7bcb9c-d24qh",
            entity_name="inventory-5c4b7bcb9c-d24qh",
        )
    ]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted)
    assert ev.score == 0.75
    assert ev.entity_grade == "服务级命中"
    assert ev.entity_resolve_method == "k8s_pod_prefix"


def test_entity_service_k8s_replicaset_prefix():
    """RS name pattern `<svc>-<10hex>` also resolves."""
    gt = [TargetEntity(entity_id="svc:cart", entity_name="cart")]
    extracted = [
        TargetEntity(entity_id="rs:cart-7c497ffb59", entity_name="cart-7c497ffb59")
    ]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted)
    assert ev.score == 0.75
    assert ev.entity_grade == "服务级命中"


def test_entity_service_reverse_direction():
    """GT is pod, agent is service → still SERVICE."""
    gt = [
        TargetEntity(
            entity_id="pod:inventory-5c4b7bcb9c-d24qh",
            entity_name="inventory-5c4b7bcb9c-d24qh",
        )
    ]
    extracted = [TargetEntity(entity_id="svc:inventory", entity_name="inventory")]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted)
    assert ev.score == 0.75


# ---------------------------------------------------------------------------
# NEIGHBOR_K via topology — 1/2/3 hop
# ---------------------------------------------------------------------------
def _make_chain_topology(*services: str) -> StubTopology:
    """Build a linear call chain: services[0] → services[1] → ... → services[-1]."""
    topo = StubTopology()
    for s in services:
        topo.add_node(f"apm@apm.service:{s}", entity_name=s, entity_type="apm.service")
    for a, b in zip(services, services[1:]):
        topo.add_edge(f"apm@apm.service:{a}", f"apm@apm.service:{b}")
    return topo


@pytest.mark.parametrize(
    "chain_len,expected_score,expected_grade",
    [
        (2, 0.40, "近邻1跳"),
        (3, 0.30, "近邻2跳"),
        (4, 0.20, "近邻3跳"),
    ],
)
def test_entity_neighbor_K_hop(chain_len, expected_score, expected_grade):
    services = [f"svc{i}" for i in range(chain_len)]
    topo = _make_chain_topology(*services)
    gt = [TargetEntity(entity_id="svc:svc0", entity_name="svc0")]
    extracted = [
        TargetEntity(
            entity_id=f"svc:svc{chain_len - 1}", entity_name=f"svc{chain_len - 1}"
        )
    ]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted, topology=topo)
    assert (
        ev.score == expected_score
    ), f"chain_len={chain_len}: expected {expected_score}, got {ev.score}"
    assert ev.entity_grade == expected_grade
    assert ev.entity_resolve_method == "topology"


def test_entity_neighbor_3hop_floor():
    """5 services apart still scores NEIGHBOR_3 ceiling; >3 hops → NONE."""
    topo = _make_chain_topology(*[f"svc{i}" for i in range(5)])
    gt = [TargetEntity(entity_id="svc:svc0", entity_name="svc0")]
    far = [TargetEntity(entity_id="svc:svc4", entity_name="svc4")]
    ev = score_entity(expected_entities=gt, extracted_entities=far, topology=topo)
    assert ev.score == 0.0
    assert ev.entity_grade == "未命中"


def test_entity_topology_string_max():
    """If string says EXACT but topology says NEIGHBOR_1, return EXACT (string wins)."""
    topo = _make_chain_topology("inventory", "cart")
    gt = [TargetEntity(entity_id="svc:inventory", entity_name="inventory")]
    extracted = [TargetEntity(entity_id="svc:inventory", entity_name="inventory")]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted, topology=topo)
    assert ev.score == 1.0
    assert ev.entity_grade == "精确命中"


def test_entity_same_service_tiebreaker():
    """Nodes in graph but disconnected, same service → SERVICE."""
    topo = StubTopology()
    topo.add_node(
        "k8s@k8s.pod:inventory-abc",
        entity_name="inventory-abc",
        belongs_to_service="inventory",
    )
    topo.add_node(
        "apm@apm.service:inventory",
        entity_name="inventory",
        belongs_to_service="inventory",
    )
    # NO edges between them
    gt = [TargetEntity(entity_id="svc:inventory", entity_name="inventory")]
    # Agent picked the pod with a unique name not matching by string EXACT/K8s prefix
    extracted = [TargetEntity(entity_id="inventory-abc", entity_name="inventory-abc")]
    ev = score_entity(expected_entities=gt, extracted_entities=extracted, topology=topo)
    # Topology finds both nodes; not connected; same service → SERVICE
    assert ev.entity_grade in (
        "服务级命中",
        "精确命中",
    )  # string SERVICE may also catch
    assert ev.score >= 0.75


# ---------------------------------------------------------------------------
# Extraction failure path — sets extraction_method for kernel penalty rule
# ---------------------------------------------------------------------------
def test_entity_llm_failed_propagates_extraction_method():
    """When extraction failed upstream, E=0 but extraction_method='llm_failed' so
    kernel does NOT apply the 0.6 F-term penalty (M0 test_entity_penalty covers downstream)."""
    ev = score_entity(
        expected_entities=[TargetEntity(entity_id="x", entity_name="x")],
        extracted_entities=[],
        extraction_method="llm_failed",
    )
    assert ev.score == 0.0
    assert ev.extraction_method == "llm_failed"


def test_entity_empty_gt_returns_zero():
    ev = score_entity(
        expected_entities=[], extracted_entities=[TargetEntity(entity_name="inventory")]
    )
    assert ev.score == 0.0
    assert ev.entity_grade == "未命中"
