from __future__ import annotations

import json
from pathlib import Path

import pytest

from brise_rca_score.llm.base import LLMClient, LLMMessage
from brise_rca_score.offline import diagnosis_from_agent_output, score_case
from brise_rca_score.scoring.mechanism import evaluate_mechanism


class QueueLLM(LLMClient):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = [json.dumps(item, ensure_ascii=False) for item in responses]
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


def _write_case(tmp_path: Path, *, roots: int = 1) -> tuple[Path, Path]:
    root = {
        "rc_id": "rc1",
        "fault_id": "F002-threadExhaustion",
        "entity_group": {
            "required_entity_ids": ["inventory-id"],
            "alias_entity_ids": [],
            "entity_refs": [
                {
                    "entity_id": "inventory-id",
                    "entity_name": "inventory",
                    "entity_domain": "apm",
                    "entity_type": "apm.service",
                }
            ],
        },
        "conclusion": "inventory worker thread pool is exhausted",
    }
    gt = {
        "ground_truth": {
            "outcome": {"root_causes": [root] * roots},
            "reasoning": {
                "steps": [
                    {
                        "step": 1,
                        "target": "inventory",
                        "fault_id": "F002-threadExhaustion",
                        "step_type": "cause",
                        "title": "线程池耗尽",
                        "description": "active threads reached pool capacity and requests were rejected",
                    }
                ]
            },
        },
        "workspace": "rca-benchmark",
        "region_id": "cn-hongkong",
        "anchor_time": "2026-07-07T09:43:47+08:00",
    }
    diagnosis = {
        "case_id": "c001",
        "extraction_status": "ok",
        "conclusion": {
            "text": "inventory has 12 blocker threads and a full queue, causing request rejection"
        },
        "fault_candidates": [
            {
                "fault_id": "",
                "raw_text": "thread pool exhausted",
                "confidence": 0.9,
                "evidence_quote": "12 blocker threads",
                "rank": 0,
            }
        ],
        "entity_candidates": [
            {
                "entity_name": "inventory",
                "confidence": 0.9,
                "evidence_quote": "inventory",
                "rank": 0,
            }
        ],
        "agent_chain": {
            "nodes": [
                {
                    "entity_name": "inventory",
                    "detail": "thread pool saturation caused rejection",
                }
            ],
            "edges": [],
        },
    }
    gt_file = tmp_path / "private-gt-bundle.json"
    diagnosis_file = tmp_path / "diagnosis.json"
    gt_file.write_text(json.dumps(gt), encoding="utf-8")
    diagnosis_file.write_text(json.dumps(diagnosis), encoding="utf-8")
    return gt_file, diagnosis_file


def test_score_existing_local_artifacts_without_trace(tmp_path: Path) -> None:
    gt_file, diagnosis_file = _write_case(tmp_path)
    llm = QueueLLM(
        [
            {
                "fault_id": "F002-threadExhaustion",
                "reason": "thread pool saturation",
            },
            {
                "fault_id": "F002-threadExhaustion",
                "reason": "thread pool saturation",
            },
            {
                "fault_id": "F002-threadExhaustion",
                "reason": "thread pool saturation",
            },
            {
                "verdict": "consistent",
                "agent_mechanism_claim": "blockers exhaust the worker pool",
                "reason": "Agent and GT describe the same mechanism",
            },
            {
                "verdict": "consistent",
                "agent_mechanism_claim": "blockers exhaust the worker pool",
                "reason": "Agent and GT describe the same mechanism",
            },
            {
                "dimensions": [
                    {
                        "dimension": "causal_contradiction",
                        "strengths": [],
                        "issues": [
                            {
                                "severity": "high",
                                "description": "one contradiction",
                                "issue_type": "mechanism_mismatch",
                                "missing": "status evidence",
                            }
                        ],
                    },
                    {
                        "dimension": "logic_jump",
                        "strengths": [],
                        "issues": [
                            {
                                "severity": "low",
                                "description": "one omitted check",
                                "issue_type": "missing_verification",
                                "missing": "direct check",
                            }
                        ],
                    },
                    {
                        "dimension": "evidence_gap",
                        "strengths": [],
                        "issues": [
                            {
                                "severity": "high",
                                "description": "missing direct metric",
                                "issue_type": "missing_metric",
                                "missing": "rejected count",
                            }
                        ],
                    },
                ],
                "reasoning": "The answer identifies the correct mechanism, but one causal statement conflicts with the expected behavior and direct metrics are missing. The path is mostly coherent while one verification step is omitted.",
            },
        ]
    )

    result = score_case(
        gt_file=gt_file,
        diagnosis_file=diagnosis_file,
        llm=llm,
    )

    assert result.scores == {
        "fault_type": 1.0,
        "entity": 1.0,
        "process": 0.5667,
        "overall": 87.0,
    }
    assert result.source["gt"] == str(gt_file)
    assert "trace" not in result.source
    assert llm.calls == 6
    assert result.details["fault_type"]["semantic_mapping_status"] == "hit"
    assert len(result.details["fault_type"]["semantic_mapping_votes"]) == 3


def test_agent_output_converts_to_score_ready_diagnosis(tmp_path: Path) -> None:
    agent_output = tmp_path / "agent_output.json"
    agent_output.write_text(
        json.dumps(
            {
                "case_id": "c001",
                "root_causes": [
                    {
                        "entity_name": "inventory",
                        "fault_type": "thread pool exhaustion",
                        "reasoning": "12 blocker threads exhausted the pool",
                        "confidence": "high",
                    }
                ],
                "causal_chain": [
                    {"cause": "thread saturation", "effect": "request rejection"}
                ],
            }
        )
    )
    diagnosis = diagnosis_from_agent_output(agent_output)
    assert diagnosis.case_id == "c001"
    assert diagnosis.fault_candidates[0].raw_text == "thread pool exhaustion"
    assert diagnosis.entity_candidates[0].entity_name == "inventory"
    assert diagnosis.agent_chain.nodes[0].detail == "request rejection"


def test_namespaced_entity_resolves_from_gt_without_remote_lookup(
    tmp_path: Path,
) -> None:
    gt_file, diagnosis_file = _write_case(tmp_path)
    diagnosis = json.loads(diagnosis_file.read_text())
    diagnosis["entity_candidates"][0]["entity_name"] = "cms-demo/inventory"
    diagnosis_file.write_text(json.dumps(diagnosis))
    replay = tmp_path / "old-score.json"
    replay.write_text(
        json.dumps(
            {
                "details": {
                    "process": {
                        "review_dimensions": [
                            {"dimension": "causal_contradiction", "issues": []},
                            {"dimension": "logic_jump", "issues": []},
                            {"dimension": "evidence_gap", "issues": []},
                        ]
                    }
                }
            }
        )
    )
    llm = QueueLLM(
        [
            {"fault_id": "F002-threadExhaustion", "reason": "same mechanism"},
            {"fault_id": "F002-threadExhaustion", "reason": "same mechanism"},
            {"fault_id": "F002-threadExhaustion", "reason": "same mechanism"},
            {"verdict": "consistent", "reason": "same mechanism"},
            {"verdict": "consistent", "reason": "same mechanism"},
        ]
    )
    result = score_case(
        gt_file=gt_file,
        diagnosis_file=diagnosis_file,
        replay_process_file=replay,
        llm=llm,
    )
    assert result.scores["entity"] == 1.0


def test_hop_discounted_fault_skips_mechanism_gate(tmp_path: Path) -> None:
    gt_file, diagnosis_file = _write_case(tmp_path)
    gt = json.loads(gt_file.read_text())
    gt["ground_truth"]["reasoning"]["steps"].append(
        {
            "step": 2,
            "target": "inventory operation",
            "fault_id": "F014-httpError5xx",
            "step_type": "impact",
            "title": "HTTP errors",
            "description": "thread rejection propagates to HTTP errors",
        }
    )
    gt_file.write_text(json.dumps(gt))
    diagnosis = json.loads(diagnosis_file.read_text())
    diagnosis["fault_candidates"][0]["fault_id"] = "F014-httpError5xx"
    diagnosis_file.write_text(json.dumps(diagnosis))
    replay = tmp_path / "brise-rca-score.json"
    replay.write_text(
        json.dumps(
            {
                "details": {
                    "process": {
                        "review_dimensions": [
                            {"dimension": "causal_contradiction", "issues": []},
                            {"dimension": "logic_jump", "issues": []},
                            {"dimension": "evidence_gap", "issues": []},
                        ]
                    }
                }
            }
        )
    )
    llm = QueueLLM([])
    result = score_case(
        gt_file=gt_file,
        diagnosis_file=diagnosis_file,
        replay_process_file=replay,
        llm=llm,
    )
    assert result.scores["fault_type"] == 0.85
    assert result.details["fault_type"]["mechanism_gate_status"] == "skipped_hop_tier"
    assert llm.calls == 0


def test_multi_root_process_replay_fails_explicitly(tmp_path: Path) -> None:
    gt_file, diagnosis_file = _write_case(tmp_path, roots=2)
    replay = tmp_path / "brise-rca-score.json"
    replay.write_text(json.dumps({"details": {"process": {}}}))
    with pytest.raises(NotImplementedError, match="per-root review evidence"):
        score_case(
            gt_file=gt_file,
            diagnosis_file=diagnosis_file,
            replay_process_file=replay,
            llm=QueueLLM([]),
        )


def test_decision_snapshot_replays_multi_root_kernel_without_llm(
    tmp_path: Path,
) -> None:
    gt_file, diagnosis_file = _write_case(tmp_path, roots=2)
    snapshot = tmp_path / "brise-rca-score.json"
    snapshot.write_text(
        json.dumps(
            {
                "details": {
                    "fault_type": {"score": 1.0, "match_tier": "exact"},
                    "entity": {
                        "score": 1.0,
                        "entity_grade": "精确命中",
                        "matched_count": 1,
                    },
                    "process": {
                        "score": 0.5667,
                        "score_source": "rca_review",
                        "review_dimensions": [],
                    },
                }
            }
        )
    )
    llm = QueueLLM([])
    result = score_case(
        gt_file=gt_file,
        diagnosis_file=diagnosis_file,
        replay_decision_file=snapshot,
        llm=llm,
    )
    assert result.scores == {
        "fault_type": 1.0,
        "entity": 1.0,
        "process": 0.5667,
        "overall": 87.0,
    }
    assert result.source["scoring"] == "decision_replay"
    assert llm.calls == 0


def test_mechanism_judge_failure_is_not_silently_accepted() -> None:
    with pytest.raises(RuntimeError, match="no valid votes"):
        evaluate_mechanism(
            llm=QueueLLM([{}, {}]),
            expected="F002-threadExhaustion",
            expected_desc="thread pool exhaustion",
            gt_mechanism="workers fill the pool",
            actual_list=["thread pool exhaustion"],
            agent_conclusion="the pool was exhausted",
            agent_chain_text="pool to timeout",
            model="test",
            temperature=0.0,
            vote_count=2,
        )


def test_runtime_modules_have_no_sls_or_brise_client_dependency() -> None:
    source_root = Path(__file__).parents[1] / "src" / "brise_rca_score"
    checked = sorted(source_root.rglob("*.py"))
    forbidden = ("TraceFetcher", "BriseClient", "sls_project", "postgres")
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
