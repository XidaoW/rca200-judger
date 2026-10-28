"""Deterministic multi-root assignment, rescue, and F/E/P/O aggregation."""

from __future__ import annotations

from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict, Field

from .models import EntityScoreEvidence, FaultTypeScoreEvidence, ProcessScoreEvidence

_QUANTUM = Decimal("0.0001")


def q4(value: object) -> Decimal:
    """Convert a score to Brise's four-decimal, half-up representation."""
    return Decimal(str(value)).quantize(_QUANTUM, rounding=ROUND_HALF_UP)


class RootInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rc_id: str
    payload: object


class AtomInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_id: str
    ordinal: int = Field(ge=0)
    payload: object


class PairEvidence(BaseModel):
    rc_id: str
    atom_id: str
    joint: Decimal
    fault: FaultTypeScoreEvidence
    entity: EntityScoreEvidence


class AssignmentEntry(BaseModel):
    rc_id: str
    atom_id: str | None = None


class RescueEvidence(BaseModel):
    rc_id: str
    atom_id: str
    score: Decimal
    selected: bool = False


class PerRootEvidence(BaseModel):
    rc_id: str
    atom_id: str | None = None
    rescue_atom_id: str | None = None
    fault: float = 0.0
    entity: float = 0.0
    process: float | None = 0.0
    process_evidence: ProcessScoreEvidence | None = None


class MultiRootResult(BaseModel):
    root_count: int
    atom_count: int
    root_count_multiplier: float
    precision_multiplier: float
    credibility_multiplier: float
    fault: float
    entity: float
    process: float | None
    overall: float
    assignment: list[AssignmentEntry]
    projected_assignment: list[AssignmentEntry]
    rescue_assignment: list[AssignmentEntry]
    pair_matrix: list[PairEvidence]
    rescue_matrix: list[RescueEvidence]
    per_root: list[PerRootEvidence]


PairScorer = Callable[
    [object, object], tuple[FaultTypeScoreEvidence, EntityScoreEvidence]
]
ProcessScorer = Callable[
    [object, object, FaultTypeScoreEvidence], ProcessScoreEvidence | None
]
RescueScorer = Callable[[object, object], float]


def _joint(fault: float, entity: float) -> Decimal:
    weighted = q4(fault) * Decimal("0.3") + q4(entity) * Decimal("0.4")
    return q4(weighted / Decimal("0.7"))


def _pair_matrix(
    roots: list[RootInput],
    atoms: list[AtomInput],
    pair_scorer: PairScorer,
) -> list[PairEvidence]:
    matrix: list[PairEvidence] = []
    for root in roots:
        for atom in atoms:
            fault, entity = pair_scorer(root.payload, atom.payload)
            matrix.append(
                PairEvidence(
                    rc_id=root.rc_id,
                    atom_id=atom.atom_id,
                    joint=_joint(fault.score, entity.score),
                    fault=fault,
                    entity=entity,
                )
            )
    return matrix


def _assignment_key(
    assignment: tuple[str | None, ...], atoms_by_id: dict[str, AtomInput]
) -> tuple[int, ...]:
    bottom = 2**63 - 1
    return tuple(
        atoms_by_id[atom_id].ordinal if atom_id is not None else bottom
        for atom_id in assignment
    )


def _maximum_assignment(
    roots: list[RootInput],
    atoms: list[AtomInput],
    weight: Callable[[str, str], Decimal],
    domains: dict[str, set[str]] | None = None,
) -> list[AssignmentEntry]:
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    best_total = Decimal("-1")
    best_assignment: tuple[str | None, ...] = (None,) * len(roots)

    def visit(index: int, used: set[str], selected: list[str | None]) -> None:
        nonlocal best_total, best_assignment
        if index == len(roots):
            assignment = tuple(selected)
            total = sum(
                (
                    weight(roots[i].rc_id, atom_id)
                    for i, atom_id in enumerate(assignment)
                    if atom_id is not None
                ),
                Decimal("0"),
            )
            if total > best_total or (
                total == best_total
                and _assignment_key(assignment, atom_by_id)
                < _assignment_key(best_assignment, atom_by_id)
            ):
                best_total = total
                best_assignment = assignment
            return
        visit(index + 1, used, [*selected, None])
        root_id = roots[index].rc_id
        allowed = domains.get(root_id, set()) if domains is not None else None
        for atom in atoms:
            if atom.atom_id in used or (
                allowed is not None and atom.atom_id not in allowed
            ):
                continue
            if weight(root_id, atom.atom_id) <= 0:
                continue
            visit(index + 1, used | {atom.atom_id}, [*selected, atom.atom_id])

    visit(0, set(), [])
    return [
        AssignmentEntry(rc_id=root.rc_id, atom_id=best_assignment[index])
        for index, root in enumerate(roots)
    ]


def _primary_assignment(
    roots: list[RootInput], atoms: list[AtomInput], matrix: list[PairEvidence]
) -> list[AssignmentEntry]:
    weights = {(cell.rc_id, cell.atom_id): cell.joint for cell in matrix}
    return _maximum_assignment(
        roots, atoms, lambda rc_id, atom_id: weights[(rc_id, atom_id)]
    )


def _project_assignment(
    roots: list[RootInput],
    atoms: list[AtomInput],
    primary: list[AssignmentEntry],
    matrix: list[PairEvidence],
    rescue_assignment: list[AssignmentEntry],
) -> list[AssignmentEntry]:
    pair_by_key = {(cell.rc_id, cell.atom_id): cell for cell in matrix}
    rescued = {entry.rc_id: entry for entry in rescue_assignment}
    used = {entry.atom_id for entry in primary if entry.atom_id is not None}
    projected: list[AssignmentEntry] = []
    for entry in primary:
        rescue_entry = rescued.get(entry.rc_id)
        if rescue_entry is not None and rescue_entry.atom_id is not None:
            projected.append(rescue_entry)
            used.add(rescue_entry.atom_id)
            continue
        if entry.atom_id is not None:
            projected.append(entry)
            continue
        unused = [atom for atom in atoms if atom.atom_id not in used]
        candidates = unused or atoms
        if not candidates:
            projected.append(entry)
            continue
        selected = max(
            candidates,
            key=lambda atom: (
                pair_by_key[(entry.rc_id, atom.atom_id)].joint,
                q4(pair_by_key[(entry.rc_id, atom.atom_id)].fault.score),
                q4(pair_by_key[(entry.rc_id, atom.atom_id)].entity.score),
                -atom.ordinal,
            ),
        )
        projected.append(AssignmentEntry(rc_id=entry.rc_id, atom_id=selected.atom_id))
        used.add(selected.atom_id)
    return projected


def _rescue_domains(
    roots: list[RootInput],
    atoms: list[AtomInput],
    primary: list[AssignmentEntry],
    matrix: list[PairEvidence],
) -> dict[str, set[str]]:
    allow_shared = len(atoms) < len(roots)
    used = {entry.atom_id for entry in primary if entry.atom_id is not None}
    pair_by_key = {(cell.rc_id, cell.atom_id): cell for cell in matrix}
    domains: dict[str, set[str]] = {}
    for entry in primary:
        if entry.atom_id is None:
            domains[entry.rc_id] = {
                atom.atom_id
                for atom in atoms
                if allow_shared or atom.atom_id not in used
            }
            continue
        if pair_by_key[(entry.rc_id, entry.atom_id)].fault.score == 0:
            domains[entry.rc_id] = {entry.atom_id}
    return domains


def _rescue(
    roots: list[RootInput],
    atoms: list[AtomInput],
    primary: list[AssignmentEntry],
    matrix: list[PairEvidence],
    rescue_scorer: RescueScorer | None,
) -> tuple[list[RescueEvidence], list[AssignmentEntry]]:
    if rescue_scorer is None:
        return [], []
    domains = _rescue_domains(roots, atoms, primary, matrix)
    root_by_id = {root.rc_id: root for root in roots}
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    rows = [
        RescueEvidence(
            rc_id=rc_id,
            atom_id=atom_id,
            score=q4(
                max(
                    0.0,
                    min(
                        0.5,
                        rescue_scorer(
                            root_by_id[rc_id].payload,
                            atom_by_id[atom_id].payload,
                        ),
                    ),
                )
            ),
        )
        for rc_id, atom_ids in domains.items()
        for atom_id in sorted(atom_ids, key=lambda item: atom_by_id[item].ordinal)
    ]
    weights = {(row.rc_id, row.atom_id): row.score for row in rows}
    domain_roots = [root for root in roots if root.rc_id in domains]
    if len(atoms) < len(roots):
        selected = [
            AssignmentEntry(
                rc_id=root.rc_id,
                atom_id=max(
                    domains[root.rc_id],
                    key=lambda atom_id: (
                        weights[(root.rc_id, atom_id)],
                        -atom_by_id[atom_id].ordinal,
                    ),
                ),
            )
            for root in domain_roots
            if domains[root.rc_id]
            and any(
                weights[(root.rc_id, atom_id)] > 0 for atom_id in domains[root.rc_id]
            )
        ]
    else:
        selected = _maximum_assignment(
            domain_roots,
            atoms,
            lambda rc_id, atom_id: weights.get((rc_id, atom_id), Decimal("0")),
            domains,
        )
        selected = [entry for entry in selected if entry.atom_id is not None]
    selected_pairs = {(entry.rc_id, entry.atom_id) for entry in selected}
    frozen = [
        row.model_copy(update={"selected": (row.rc_id, row.atom_id) in selected_pairs})
        for row in rows
    ]
    return frozen, selected


def _precision(root_count: int, atom_count: int) -> Decimal:
    if atom_count <= root_count:
        return Decimal("1")
    return q4((Decimal(root_count) / Decimal(atom_count)).sqrt())


def _root_count_multiplier(root_count: int, atom_count: int) -> Decimal:
    return q4(min(Decimal("1"), Decimal(atom_count) / Decimal(root_count)))


def _aggregate_process(per_root: list[PerRootEvidence]) -> Decimal | None:
    available = [item.process for item in per_root if item.process is not None]
    if not available:
        return None
    return q4(sum((q4(value) for value in available), Decimal("0")) / len(available))


def _aggregate(
    per_root: list[PerRootEvidence],
    *,
    root_count: int,
    atom_count: int,
    credibility_multiplier: float,
) -> tuple[Decimal, Decimal, Decimal | None, Decimal]:
    precision = _precision(root_count, atom_count)
    count_multiplier = _root_count_multiplier(root_count, atom_count)
    fault_mean = sum((q4(item.fault) for item in per_root), Decimal("0")) / root_count
    fault = q4(fault_mean * precision * count_multiplier)
    entity_mean = sum((q4(item.entity) for item in per_root), Decimal("0")) / root_count
    if atom_count > 2 * root_count:
        entity_mean *= precision
    entity = q4(entity_mean * count_multiplier)
    process_mean = _aggregate_process(per_root)
    process = q4(process_mean * count_multiplier) if process_mean is not None else None
    if process is None:
        raw = (fault * Decimal("0.3") + entity * Decimal("0.4")) / Decimal("0.7")
    else:
        raw = (
            fault * Decimal("0.3") + entity * Decimal("0.4") + process * Decimal("0.3")
        )
    return fault, entity, process, q4(raw * q4(credibility_multiplier))


def compute_multi_root(
    *,
    roots: list[RootInput],
    atoms: list[AtomInput],
    pair_scorer: PairScorer,
    process_scorer: ProcessScorer,
    rescue_scorer: RescueScorer | None = None,
    credibility_multiplier: float = 1.0,
) -> MultiRootResult:
    """Score multi-root claims with deterministic assignment then bounded rescue."""
    if not 2 <= len(roots) <= 3:
        raise ValueError("multi-root scoring requires 2 or 3 root causes")
    if len({root.rc_id for root in roots}) != len(roots):
        raise ValueError("rc_id must be unique")
    if len({atom.atom_id for atom in atoms}) != len(atoms):
        raise ValueError("atom_id must be unique")
    if len({atom.ordinal for atom in atoms}) != len(atoms):
        raise ValueError("atom ordinal must be unique")
    ordered_roots = sorted(roots, key=lambda item: item.rc_id)
    ordered_atoms = sorted(atoms, key=lambda item: item.ordinal)
    matrix = _pair_matrix(ordered_roots, ordered_atoms, pair_scorer)
    primary = _primary_assignment(ordered_roots, ordered_atoms, matrix)
    rescue_matrix, rescue_assignment = _rescue(
        ordered_roots,
        ordered_atoms,
        primary,
        matrix,
        rescue_scorer,
    )
    projected = _project_assignment(
        ordered_roots,
        ordered_atoms,
        primary,
        matrix,
        rescue_assignment,
    )
    root_by_id = {root.rc_id: root for root in ordered_roots}
    atom_by_id = {atom.atom_id: atom for atom in ordered_atoms}
    pair_by_key = {(cell.rc_id, cell.atom_id): cell for cell in matrix}
    rescue_by_root = {entry.rc_id: entry.atom_id for entry in rescue_assignment}
    rescue_score = {
        (row.rc_id, row.atom_id): row.score for row in rescue_matrix if row.selected
    }
    per_root: list[PerRootEvidence] = []
    for entry in projected:
        if entry.atom_id is None:
            per_root.append(PerRootEvidence(rc_id=entry.rc_id))
            continue
        pair = pair_by_key[(entry.rc_id, entry.atom_id)]
        rescue_atom_id = rescue_by_root.get(entry.rc_id)
        rescued = rescue_score.get((entry.rc_id, rescue_atom_id or ""), Decimal("0"))
        effective_fault = max(q4(pair.fault.score), rescued)
        process_atom_id = entry.atom_id if pair.fault.score > 0 else rescue_atom_id
        process_evidence: ProcessScoreEvidence | None = ProcessScoreEvidence(
            score=0.0,
            score_source="rca_review",
        )
        if process_atom_id is not None:
            process_evidence = process_scorer(
                root_by_id[entry.rc_id].payload,
                atom_by_id[process_atom_id].payload,
                pair.fault.model_copy(update={"score": float(effective_fault)}),
            )
        per_root.append(
            PerRootEvidence(
                rc_id=entry.rc_id,
                atom_id=entry.atom_id,
                rescue_atom_id=rescue_atom_id,
                fault=float(effective_fault),
                entity=pair.entity.score,
                process=(
                    process_evidence.score if process_evidence is not None else None
                ),
                process_evidence=process_evidence,
            )
        )
    fault, entity, process, composite = _aggregate(
        per_root,
        root_count=len(ordered_roots),
        atom_count=len(ordered_atoms),
        credibility_multiplier=credibility_multiplier,
    )
    return MultiRootResult(
        root_count=len(ordered_roots),
        atom_count=len(ordered_atoms),
        root_count_multiplier=float(
            _root_count_multiplier(len(ordered_roots), len(ordered_atoms))
        ),
        precision_multiplier=float(_precision(len(ordered_roots), len(ordered_atoms))),
        credibility_multiplier=credibility_multiplier,
        fault=float(fault),
        entity=float(entity),
        process=float(process) if process is not None else None,
        overall=float(q4(composite * Decimal("100"))),
        assignment=primary,
        projected_assignment=projected,
        rescue_assignment=rescue_assignment,
        pair_matrix=matrix,
        rescue_matrix=rescue_matrix,
        per_root=per_root,
    )
