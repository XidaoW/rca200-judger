"""Local JSON topology adapter used by entity scoring.

Supported payloads contain ``entities`` and ``edges`` arrays. Entity records use
``id``, ``type`` and ``name``. Edge records may use either
``src``/``dst``/``relation`` or ``source``/``target``/``type``.

The adapter builds an in-memory graph for name resolution, structural distance,
service ancestry and host-node matching. It implements the ``TopologyGraph``
protocol defined in :mod:`brise_rca_score.topology.base`.

Load a local graph with::

    JsonTopology.from_file("cases/c001/topology.json")
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .base import GraphNode

# Edge relations that contribute to the hierarchical "same service" / "same node" check
_CONTAINS_LINK_TYPES = {"contains"}
_HOSTS_LINK_TYPES = {"hosts"}
# Entity types we treat as "service-level" anchors for ancestry climbing
_SERVICE_ANCHOR_TYPES = {"apm.service", "k8s.service"}


@dataclass
class JsonTopology:
    """In-memory graph from one case's topology.json.

    Public surface matches the TopologyGraph Protocol (4 methods used by
    brise_rca_score.scoring.entity._topology_distance_match).
    """

    case_id: str = ""
    nodes_by_id: dict[str, GraphNode] = field(default_factory=dict)
    name_index: dict[str, str] = field(default_factory=dict)  # lower(name) → id
    # Undirected adjacency over all available edge relations (used for BFS)
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    # Reverse hierarchy via `contains` (used for same_service ancestry)
    contained_in: dict[str, set[str]] = field(
        default_factory=dict
    )  # entity_id → parents
    contains_children: dict[str, set[str]] = field(
        default_factory=dict
    )  # entity_id → children
    # Hosts relation: k8s.pod --hosts--> apm.instance.
    # We store both directions to walk service↔instance↔pod↔node chains.
    instance_of_pod: dict[str, set[str]] = field(
        default_factory=dict
    )  # pod → {instance}
    pod_of_instance: dict[str, set[str]] = field(
        default_factory=dict
    )  # instance → {pod}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def from_dict(payload: dict) -> "JsonTopology":
        topo = JsonTopology(case_id=payload.get("case_id", ""))
        for ent in payload.get("entities") or []:
            eid = ent.get("id", "")
            name = ent.get("name", "")
            etype = ent.get("type", "")
            if not eid:
                continue
            topo.nodes_by_id[eid] = GraphNode(
                composite_id=eid,
                entity_id=eid,
                entity_name=name,
                entity_domain=etype.split(".", 1)[0] if "." in etype else "",
                entity_type=etype,
            )
            if name:
                # If multiple entities share a name, first one wins (matches brise behavior)
                topo.name_index.setdefault(name.strip().lower(), eid)

        for edge in payload.get("edges") or []:
            src = edge.get("src") or edge.get("source") or ""
            dst = edge.get("dst") or edge.get("target") or ""
            rel = edge.get("relation") or edge.get("type") or ""
            if not src or not dst:
                continue
            # Brise entity_distance traverses all topology link types.
            topo.adjacency.setdefault(src, set()).add(dst)
            topo.adjacency.setdefault(dst, set()).add(src)
            if rel in _CONTAINS_LINK_TYPES:
                # `contains` is directional: src CONTAINS dst.
                topo.contained_in.setdefault(dst, set()).add(src)
                topo.contains_children.setdefault(src, set()).add(dst)
            if rel in _HOSTS_LINK_TYPES:
                # k8s.pod (src) --hosts--> apm.instance (dst)
                topo.instance_of_pod.setdefault(src, set()).add(dst)
                topo.pod_of_instance.setdefault(dst, set()).add(src)

        return topo

    @staticmethod
    def from_file(path: str | Path) -> "JsonTopology":
        return JsonTopology.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    # ------------------------------------------------------------------
    # TopologyGraph Protocol
    # ------------------------------------------------------------------
    def find_node_by_name(self, name: str) -> str | None:
        if not name:
            return None
        key = name.strip().lower()
        # Direct id match (callers may pass entity_id)
        if name in self.nodes_by_id:
            return name
        return self.name_index.get(key)

    def structural_distance(self, a: str, b: str) -> int | None:
        if a == b:
            return 0
        if a not in self.nodes_by_id or b not in self.nodes_by_id:
            return None
        # BFS over `calls` adjacency
        visited = {a}
        q = deque([(a, 0)])
        while q:
            cur, depth = q.popleft()
            if cur == b:
                return depth
            if depth >= 10:  # safety; brise uses max_depth=3
                continue
            for nb in self.adjacency.get(cur, ()):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, depth + 1))
        return None

    def get_node(self, composite_id: str) -> GraphNode | None:
        return self.nodes_by_id.get(composite_id)

    def same_service(self, a: str, b: str) -> bool:
        """Both nodes resolve to the same apm.service ancestor via `contains` chain.

        Walks the contained_in tree from each node up to the first
        apm.service/k8s.service ancestor; same ancestor → same service.
        """
        svc_a = self._climb_to_service(a)
        svc_b = self._climb_to_service(b)
        return bool(svc_a) and svc_a == svc_b

    # ------------------------------------------------------------------
    # HOST_NODE grade derived from explicit `hosts` and `contains` edges.
    # ------------------------------------------------------------------
    def host_nodes_of(self, entity_id: str) -> set[str]:
        """Return all k8s.node ids that ultimately host this entity.

        Supported hierarchy chains:
            k8s.node               → self
            k8s.pod                → its containing k8s.node (via contains)
            apm.instance           → pod(s) that host it → their nodes
            apm.service            → its contained instances → ... → nodes
            apm.operation          → climb to containing service → ...
        """
        if not entity_id or entity_id not in self.nodes_by_id:
            return set()
        node = self.nodes_by_id[entity_id]
        t = node.entity_type
        out: set[str] = set()

        if t == "k8s.node":
            return {entity_id}

        if t == "k8s.pod":
            for parent in self.contained_in.get(entity_id, ()):
                pnode = self.nodes_by_id.get(parent)
                if pnode and pnode.entity_type == "k8s.node":
                    out.add(parent)
            return out

        if t == "apm.instance":
            for pod in self.pod_of_instance.get(entity_id, ()):
                out |= self.host_nodes_of(pod)
            return out

        if t in ("apm.service", "k8s.service"):
            # contained: service --contains--> instance/operation/pod
            # We walk *children* (entities whose parents include this id)
            for child in self._children_via_contains(entity_id):
                ctype = self.nodes_by_id.get(child, GraphNode("")).entity_type
                if ctype in ("apm.instance", "k8s.pod"):
                    out |= self.host_nodes_of(child)
            return out

        # operation / external / etc: climb to containing service
        for parent in self.contained_in.get(entity_id, ()):
            out |= self.host_nodes_of(parent)
        return out

    def host_node_of(self, entity_id: str) -> str | None:
        """First k8s.node id reached (compat alias)."""
        s = self.host_nodes_of(entity_id)
        return next(iter(s)) if s else None

    def same_host_node(self, a: str, b: str) -> bool:
        na = self.host_nodes_of(a)
        nb = self.host_nodes_of(b)
        return bool(na & nb)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _children_via_contains(self, eid: str) -> set[str]:
        return self.contains_children.get(eid, set())

    def _climb_to_service(self, eid: str, _seen: set[str] | None = None) -> str | None:
        if eid not in self.nodes_by_id:
            return None
        node = self.nodes_by_id[eid]
        if node.entity_type in _SERVICE_ANCHOR_TYPES:
            return eid
        _seen = _seen or set()
        _seen.add(eid)
        for parent in self.contained_in.get(eid, ()):
            if parent in _seen:
                continue
            svc = self._climb_to_service(parent, _seen)
            if svc:
                return svc
        # Raw entity properties are not retained, so no service fallback is available.
        return None
