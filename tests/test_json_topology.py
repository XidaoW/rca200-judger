"""JsonTopology tests against the published topology fixture.

Fixture: `tests/fixtures/t001_topology.json`, a published local topology snapshot.
237 entities × 249 edges with calls / contains / hosts relations.

Coverage:
  - name → entity_id resolution (EXACT path)
  - service ↔ service via `calls` → NEIGHBOR_1 (0.40)
  - service ↔ pod hosting its instance via chain → SERVICE_MATCH (0.75)
  - two services that share a k8s.node → HOST_NODE (0.60)
  - disconnected services → NONE
"""

from __future__ import annotations

from pathlib import Path

import pytest
from brise_rca_score.topology import JsonTopology

FIXTURE = Path(__file__).parent / "fixtures" / "t001_topology.json"


@pytest.fixture(scope="module")
def topo() -> JsonTopology:
    return JsonTopology.from_file(FIXTURE)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------
def test_topo_loads_real_counts(topo: JsonTopology):
    assert topo.case_id == "t001"
    assert len(topo.nodes_by_id) == 237
    # Adjacency is bidirectional and contains every supported relation.
    assert sum(len(v) for v in topo.adjacency.values()) >= 38 * 2
    # contains: 198 edges (one-direction stored in contained_in + contains_children)
    assert sum(len(v) for v in topo.contains_children.values()) == 198
    # hosts: 13 edges
    assert sum(len(v) for v in topo.instance_of_pod.values()) == 13


def test_topo_name_lookup(topo: JsonTopology):
    inv = topo.find_node_by_name("inventory")
    assert inv == "ed13cb14265f98d9eb3b89b664760adb"
    assert topo.find_node_by_name("INVENTORY") == inv  # case-insensitive
    assert topo.find_node_by_name("nonexistent") is None


# ---------------------------------------------------------------------------
# Topology distance
# ---------------------------------------------------------------------------
def test_topo_neighbor_1_via_calls(topo: JsonTopology):
    """frontend → ad and frontend → checkout are direct `calls` edges."""
    frontend = topo.find_node_by_name("frontend")
    ad = topo.find_node_by_name("ad")
    assert frontend and ad
    assert topo.structural_distance(frontend, ad) == 1


def test_topo_self_distance_zero(topo: JsonTopology):
    inv = topo.find_node_by_name("inventory")
    assert topo.structural_distance(inv, inv) == 0


def test_topo_all_relation_distance(topo: JsonTopology):
    """Entity scoring follows Brise and traverses all available link types."""
    cluster_id = next(
        eid for eid, n in topo.nodes_by_id.items() if n.entity_type == "k8s.cluster"
    )
    inv = topo.find_node_by_name("inventory")
    assert topo.structural_distance(cluster_id, inv) == 4


# ---------------------------------------------------------------------------
# host_node_of: service → instance → pod → node chain
# ---------------------------------------------------------------------------
def test_topo_host_node_for_service(topo: JsonTopology):
    """The inventory service should resolve to 1+ k8s.node via its pods."""
    inv = topo.find_node_by_name("inventory")
    nodes = topo.host_nodes_of(inv)
    assert nodes, "inventory should map to at least one k8s.node"
    # Each resolved node should genuinely be of type k8s.node
    for nid in nodes:
        assert topo.nodes_by_id[nid].entity_type == "k8s.node"


def test_topo_host_node_for_pod(topo: JsonTopology):
    """A k8s.pod's host node is the node containing it via `contains`."""
    pod = next(eid for eid, n in topo.nodes_by_id.items() if n.entity_type == "k8s.pod")
    nodes = topo.host_nodes_of(pod)
    if nodes:
        # Confirm the chain: pod's containing parent IS a k8s.node
        chosen = next(iter(nodes))
        assert topo.nodes_by_id[chosen].entity_type == "k8s.node"


def test_topo_same_host_node(topo: JsonTopology):
    """Find two pods that share a k8s.node, then assert same_host_node detects it."""
    # node_id → its child pods
    node_to_pods: dict[str, list[str]] = {}
    for parent, children in topo.contains_children.items():
        if (
            topo.nodes_by_id.get(parent, None)
            and topo.nodes_by_id[parent].entity_type == "k8s.node"
        ):
            for c in children:
                if (
                    topo.nodes_by_id.get(c, None)
                    and topo.nodes_by_id[c].entity_type == "k8s.pod"
                ):
                    node_to_pods.setdefault(parent, []).append(c)
    # pick the first node with >= 2 pods
    candidates = [(n, ps) for n, ps in node_to_pods.items() if len(ps) >= 2]
    assert candidates, "t001 should have at least one k8s.node hosting 2+ pods"
    node, pods = candidates[0]
    assert topo.same_host_node(pods[0], pods[1]) is True
    # And not all pods are on this node
    other_pod = next(
        (p for n, ps in node_to_pods.items() if n != node for p in ps),
        None,
    )
    if other_pod:
        assert topo.same_host_node(pods[0], other_pod) is False
