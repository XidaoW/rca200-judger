from __future__ import annotations

import json
from pathlib import Path

import pytest
from brise_rca_score.llm.base import LLMClient, LLMMessage
from brise_rca_score.offline import score_case
from brise_rca_score.scoring.models import (
    EntityScoreEvidence,
    FaultTypeScoreEvidence,
    ProcessScoreEvidence,
)
from brise_rca_score.scoring.multi_root import (
    AtomInput,
    RootInput,
    compute_multi_root,
)


class QueueLLM(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = [json.dumps(item) for item in responses]
        self.calls = 0

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "text",
    ) -> str:
        del messages, model, temperature, max_tokens, response_format
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _pair(root: object, atom: object):
    root_id = str(root)
    atom_id = str(atom)
    hit = root_id[-1] == atom_id[-1]
    score = 1.0 if hit else 0.0
    return (
        FaultTypeScoreEvidence(score=score, match_tier="exact" if hit else "none"),
        EntityScoreEvidence(
            score=score,
            entity_grade="精确命中" if hit else "未命中",
        ),
    )


def _process(root: object, atom: object, fault: FaultTypeScoreEvidence):
    del root, atom, fault
    return ProcessScoreEvidence(score=0.7, score_source="rca_review")


def test_two_roots_two_atoms_assign_exactly() -> None:
    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[
            AtomInput(atom_id="a1", ordinal=0, payload="a1"),
            AtomInput(atom_id="a2", ordinal=1, payload="a2"),
        ],
        pair_scorer=_pair,
        process_scorer=_process,
    )
    assert [(item.rc_id, item.atom_id) for item in result.assignment] == [
        ("rc1", "a1"),
        ("rc2", "a2"),
    ]
    assert result.fault == 1.0
    assert result.entity == 1.0
    assert result.process == 0.7
    assert result.overall == 91.0


def test_missing_root_is_penalized_by_equal_weight_average() -> None:
    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[AtomInput(atom_id="a1", ordinal=0, payload="a1")],
        pair_scorer=_pair,
        process_scorer=_process,
    )
    assert result.root_count_multiplier == 0.5
    assert result.fault == 0.25
    assert result.entity == 0.25
    assert result.process == 0.175
    assert result.overall == 22.75


def test_multi_root_applies_credibility_multiplier() -> None:
    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[
            AtomInput(atom_id="a1", ordinal=0, payload="a1"),
            AtomInput(atom_id="a2", ordinal=1, payload="a2"),
        ],
        pair_scorer=_pair,
        process_scorer=_process,
        credibility_multiplier=0.5,
    )
    assert result.credibility_multiplier == 0.5
    assert result.overall == 45.5


def test_process_na_reweights_only_available_roots() -> None:
    def process_with_na(root: object, atom: object, fault: FaultTypeScoreEvidence):
        del atom, fault
        if root == "r2":
            return None
        return ProcessScoreEvidence(score=0.7, score_source="rca_review")

    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[
            AtomInput(atom_id="a1", ordinal=0, payload="a1"),
            AtomInput(atom_id="a2", ordinal=1, payload="a2"),
        ],
        pair_scorer=_pair,
        process_scorer=process_with_na,
    )
    assert result.process == 0.7
    assert result.overall == 91.0


def test_shared_rescue_is_separate_from_primary_assignment() -> None:
    def unmatched_pair(root: object, atom: object):
        del root, atom
        return (
            FaultTypeScoreEvidence(score=0.0, match_tier="none"),
            EntityScoreEvidence(score=1.0, entity_grade="精确命中"),
        )

    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[AtomInput(atom_id="a1", ordinal=0, payload="a1")],
        pair_scorer=unmatched_pair,
        rescue_scorer=lambda root, atom: 0.5,
        process_scorer=_process,
    )
    assert result.assignment[0].atom_id == "a1"
    assert [entry.atom_id for entry in result.rescue_assignment] == ["a1", "a1"]
    assert result.fault == 0.25
    assert result.entity == 0.5
    assert result.process == 0.35
    assert result.overall == 38.0


def test_selected_shared_rescue_drives_projected_assignment() -> None:
    def pair(root: object, atom: object):
        hit = (root, atom) in {("r1", "a1"), ("r2", "a2")}
        score = 1.0 if hit else 0.0
        return (
            FaultTypeScoreEvidence(score=score, match_tier="exact" if hit else "none"),
            EntityScoreEvidence(
                score=score,
                entity_grade="精确命中" if hit else "未命中",
            ),
        )

    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
            RootInput(rc_id="rc3", payload="r3"),
        ],
        atoms=[
            AtomInput(atom_id="a1", ordinal=0, payload="a1"),
            AtomInput(atom_id="a2", ordinal=1, payload="a2"),
        ],
        pair_scorer=pair,
        rescue_scorer=lambda root, atom: 0.5 if (root, atom) == ("r3", "a2") else 0.0,
        process_scorer=_process,
    )

    projected = {entry.rc_id: entry.atom_id for entry in result.projected_assignment}
    rescued = {entry.rc_id: entry.atom_id for entry in result.rescue_assignment}
    assert rescued["rc3"] == "a2"
    assert projected["rc3"] == "a2"


def test_projection_tie_prefers_fault_then_entity_before_ordinal() -> None:
    def tied_pair(root: object, atom: object):
        if (root, atom) in {("r1", "a1"), ("r2", "a2")}:
            fault = entity = 1.0
        elif (root, atom) == ("r3", "a1"):
            fault, entity = 0.0, 0.75
        elif (root, atom) == ("r3", "a2"):
            fault, entity = 1.0, 0.0
        else:
            fault = entity = 0.0
        return (
            FaultTypeScoreEvidence(
                score=fault, match_tier="exact" if fault else "none"
            ),
            EntityScoreEvidence(
                score=entity,
                entity_grade="精确命中" if entity else "未命中",
            ),
        )

    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
            RootInput(rc_id="rc3", payload="r3"),
        ],
        atoms=[
            AtomInput(atom_id="a1", ordinal=0, payload="a1"),
            AtomInput(atom_id="a2", ordinal=1, payload="a2"),
        ],
        pair_scorer=tied_pair,
        process_scorer=_process,
    )

    projected = {entry.rc_id: entry.atom_id for entry in result.projected_assignment}
    assert projected["rc3"] == "a2"


def test_empty_atoms_return_zero_scores_without_calling_rescue() -> None:
    def unexpected_rescue(root: object, atom: object) -> float:
        raise AssertionError(f"unexpected rescue call: {root}:{atom}")

    result = compute_multi_root(
        roots=[
            RootInput(rc_id="rc1", payload="r1"),
            RootInput(rc_id="rc2", payload="r2"),
        ],
        atoms=[],
        pair_scorer=_pair,
        rescue_scorer=unexpected_rescue,
        process_scorer=_process,
    )

    assert result.root_count_multiplier == 0.0
    assert result.fault == 0.0
    assert result.entity == 0.0
    assert result.process == 0.0
    assert result.overall == 0.0
    assert all(entry.atom_id is None for entry in result.projected_assignment)


def test_duplicate_atom_ordinals_are_rejected() -> None:
    with pytest.raises(ValueError, match="atom ordinal must be unique"):
        compute_multi_root(
            roots=[
                RootInput(rc_id="rc1", payload="r1"),
                RootInput(rc_id="rc2", payload="r2"),
            ],
            atoms=[
                AtomInput(atom_id="a1", ordinal=0, payload="a1"),
                AtomInput(atom_id="a2", ordinal=0, payload="a2"),
            ],
            pair_scorer=_pair,
            process_scorer=_process,
        )


def _process_response() -> dict[str, object]:
    return {
        "dimensions": [
            {
                "dimension": name,
                "strengths": [],
                "issues": [{"severity": "low", "description": "minor gap"}],
            }
            for name in ("causal_contradiction", "logic_jump", "evidence_gap")
        ],
        "reasoning": "The root-cause path is materially correct and evidence is present, but each dimension has one minor completeness gap that does not change the final direction.",
    }


def test_live_multi_root_scores_agent_output(tmp_path: Path) -> None:
    gt = {
        "ground_truth": {
            "outcome": {
                "root_causes": [
                    {
                        "rc_id": "rc1",
                        "fault_id": "F014-httpError5xx",
                        "entity_group": {
                            "entity_refs": [
                                {"entity_id": "e1", "entity_name": "invoicing"}
                            ]
                        },
                        "conclusion": "invoicing returns 5xx",
                    },
                    {
                        "rc_id": "rc2",
                        "fault_id": "F034-cpuFullLoad",
                        "entity_group": {
                            "entity_refs": [
                                {"entity_id": "e2", "entity_name": "statement"}
                            ]
                        },
                        "conclusion": "statement CPU reaches its limit",
                    },
                ]
            },
            "reasoning": {
                "steps": [
                    {
                        "step": 1,
                        "rc_refs": ["rc1"],
                        "fault_id": "F014-httpError5xx",
                        "step_type": "cause",
                        "target": "invoicing",
                        "description": "invoicing returns 5xx",
                    },
                    {
                        "step": 2,
                        "rc_refs": ["rc2"],
                        "fault_id": "F034-cpuFullLoad",
                        "step_type": "cause",
                        "target": "statement",
                        "description": "statement CPU reaches its limit",
                    },
                ]
            },
        }
    }
    agent = {
        "case_id": "c022",
        "root_causes": [
            {
                "entity_name": "invoicing",
                "reasoning": "invoicing returns HTTP 500 for each request",
                "confidence": "high",
            },
            {
                "entity_name": "statement",
                "reasoning": "statement CPU is pinned at 100 percent",
                "confidence": "high",
            },
        ],
    }
    gt_file = tmp_path / "private-gt-bundle.json"
    agent_file = tmp_path / "agent_output.json"
    gt_file.write_text(json.dumps(gt))
    agent_file.write_text(json.dumps(agent))
    llm = QueueLLM(
        [
            {"fault_id": "F014-httpError5xx", "reason": "HTTP 5xx mechanism"},
            {"fault_id": "F014-httpError5xx", "reason": "HTTP 5xx mechanism"},
            {"fault_id": "F014-httpError5xx", "reason": "HTTP 5xx mechanism"},
            {"fault_id": "F034-cpuFullLoad", "reason": "CPU saturation"},
            {"fault_id": "F034-cpuFullLoad", "reason": "CPU saturation"},
            {"fault_id": "F034-cpuFullLoad", "reason": "CPU saturation"},
            _process_response(),
            _process_response(),
        ]
    )
    result = score_case(
        gt_file=gt_file,
        agent_output_file=agent_file,
        llm=llm,
    )
    assert result.scores == {
        "fault_type": 1.0,
        "entity": 1.0,
        "process": 0.7,
        "overall": 91.0,
    }
    assert result.source["scoring"] == "dashscope_multi_root"
    assert llm.calls == 8
