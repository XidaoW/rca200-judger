"""Topology abstractions used by entity scoring.

The scoring kernel depends on a small ``TopologyGraph`` protocol rather than a
remote topology service. Production scoring can load a local JSON graph, tests
can use ``StubTopology``, and callers without a graph retain string matching.

Required graph operations are name resolution, node lookup, structural distance
and same-service detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GraphNode:
    """Minimal topology node, mirroring brise EntityNode fields used by scoring."""

    composite_id: (
        str  # e.g. "apm@apm.service:inventory" or "k8s@k8s.pod:inventory-abc-def"
    )
    entity_id: str = ""  # raw id (without domain@type prefix)
    entity_name: str = ""  # display / service name
    entity_domain: str = ""  # apm | k8s | acs
    entity_type: str = ""  # apm.service | k8s.pod | k8s.node | ...


@dataclass
class GraphEdge:
    source_id: str  # composite_id
    target_id: str  # composite_id
    link_type: str = "calls"  # calls | contains | runs_on | ...


class TopologyGraph(Protocol):
    """Minimal protocol — implement these 4 methods to plug a graph backend."""

    def find_node_by_name(self, name: str) -> str | None:
        """Return composite_id of the node whose entity_name or entity_id matches `name`.

        Match is case-insensitive. Returns None if not found.
        """
        ...

    def structural_distance(self, a: str, b: str) -> int | None:
        """BFS hop count between two composite_ids over default link types.

        Returns 0 when a == b. Returns None when disconnected.
        """
        ...

    def get_node(self, composite_id: str) -> GraphNode | None:
        """Look up a node by composite_id."""
        ...

    def same_service(self, a: str, b: str) -> bool:
        """Whether two nodes belong to the same service (e.g., pod a + svc b).

        Brise uses this as a tie-breaker when two nodes are in the graph but not
        BFS-connected via calls edges.
        """
        ...


# ---------------------------------------------------------------------------
# StubTopology — in-memory test fixture matching the Protocol
# ---------------------------------------------------------------------------
@dataclass
class StubTopology:
    """Hand-crafted graph for tests. NOT a Protocol method; for ergonomics:
    topo = StubTopology()
    topo.add_node("apm@apm.service:inventory", entity_name="inventory")
    topo.add_node("apm@apm.service:cart", entity_name="cart")
    topo.add_edge("cart→inventory", source="apm@apm.service:cart", target="apm@apm.service:inventory")
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    # service_for[composite_id] → service_name (drives same_service tie-breaker)
    service_for: dict[str, str] = field(default_factory=dict)

    def add_node(
        self,
        composite_id: str,
        *,
        entity_name: str = "",
        entity_id: str = "",
        entity_domain: str = "",
        entity_type: str = "",
        belongs_to_service: str = "",
    ) -> None:
        self.nodes[composite_id] = GraphNode(
            composite_id=composite_id,
            entity_id=entity_id or composite_id.rsplit(":", 1)[-1],
            entity_name=entity_name,
            entity_domain=entity_domain,
            entity_type=entity_type,
        )
        if belongs_to_service:
            self.service_for[composite_id] = belongs_to_service

    def add_edge(self, source: str, target: str, link_type: str = "calls") -> None:
        self.edges.append(
            GraphEdge(source_id=source, target_id=target, link_type=link_type)
        )

    # Protocol methods --------------------------------------------------
    def find_node_by_name(self, name: str) -> str | None:
        key = (name or "").strip().lower()
        if not key:
            return None
        for cid, node in self.nodes.items():
            if cid.lower() == key:
                return cid
            if node.entity_name.lower() == key:
                return cid
            if node.entity_id.lower() == key:
                return cid
        return None

    def structural_distance(self, a: str, b: str) -> int | None:
        if a == b:
            return 0
        if a not in self.nodes or b not in self.nodes:
            return None
        # bidirectional BFS over `calls` edges (brise default link_type)
        from collections import deque

        # build adjacency (undirected for distance — brise tries both directions)
        adj: dict[str, set[str]] = {cid: set() for cid in self.nodes}
        for e in self.edges:
            if e.link_type != "calls":
                continue
            if e.source_id in adj and e.target_id in adj:
                adj[e.source_id].add(e.target_id)
                adj[e.target_id].add(e.source_id)

        visited = {a}
        q = deque([(a, 0)])
        while q:
            node, depth = q.popleft()
            if node == b:
                return depth
            if depth >= 10:  # safety bound; brise uses max_depth=3
                continue
            for nb in adj.get(node, ()):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, depth + 1))
        return None

    def get_node(self, composite_id: str) -> GraphNode | None:
        return self.nodes.get(composite_id)

    def same_service(self, a: str, b: str) -> bool:
        sa = self.service_for.get(a, "")
        sb = self.service_for.get(b, "")
        return bool(sa) and sa == sb
