from __future__ import annotations

import json

from brise_rca_score.llm.base import LLMClient, LLMMessage
from brise_rca_score.scoring.mechanism import mechanism_gate_prompt_sha256
from brise_rca_score.scoring.semantic_map import (
    map_unresolved_fault_text,
    parse_semantic_map_vote,
    semantic_map_prompt_sha256,
)


class QueueLLM(LLMClient):
    def __init__(self, responses: list[str | dict[str, object]]) -> None:
        self.responses = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in responses
        ]
        self.calls: list[list[LLMMessage]] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "text",
    ) -> str:
        del model, temperature, max_tokens, response_format
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


def _dictionary() -> dict[str, str]:
    return {
        "f010-slowsql": "F010-slowSQL",
        "slowsql": "F010-slowSQL",
        "f161-rdsinefficientquerystorm": "F161-rdsInefficientQueryStorm",
        "rdsinefficientquerystorm": "F161-rdsInefficientQueryStorm",
    }


def test_active_semantic_map_prompt_snapshot_sha() -> None:
    assert semantic_map_prompt_sha256() == (
        "56df534f335afa25091f6b775f09a1148fbb2f04dd6dc6c0be12b7f3886e8278"
    )


def test_active_mechanism_gate_prompt_snapshot_sha() -> None:
    assert mechanism_gate_prompt_sha256() == (
        "f42e4ccf8e8dbf89fe3e422414d6995042810b499c2f14c22e0733d72095a167"
    )


def test_semantic_map_uses_majority_and_normalizes_aliases() -> None:
    llm = QueueLLM(
        [
            {"fault_id": "slowSQL", "reason": "query latency"},
            {"fault_id": "F010-slowSQL", "reason": "same mechanism"},
            {"fault_id": "F161-rdsInefficientQueryStorm", "reason": "minority"},
        ]
    )
    result = map_unresolved_fault_text(
        llm=llm,
        raw_text="slow queries block the application",
        reference="- F010-slowSQL: slow SQL",
        dictionary=_dictionary(),
        context="RDS application context",
        vote_count=3,
    )
    assert result.status == "hit"
    assert result.fault_id == "F010-slowSQL"
    assert [vote.fault_id if vote else None for vote in result.votes] == [
        "F010-slowSQL",
        "F010-slowSQL",
        "F161-rdsInefficientQueryStorm",
    ]
    assert len(llm.calls) == 3
    assert "主句机制优先" in llm.calls[0][0].content
    assert "RDS application context" in llm.calls[0][0].content


def test_semantic_map_blank_majority_is_real_miss() -> None:
    result = map_unresolved_fault_text(
        llm=QueueLLM(
            [
                {"fault_id": "", "reason": "no mechanism match"},
                {"fault_id": "", "reason": "insufficient mechanism"},
                {"fault_id": "F010-slowSQL", "reason": "minority"},
            ]
        ),
        raw_text="unable to determine the cause",
        reference="- F010-slowSQL: slow SQL",
        dictionary=_dictionary(),
        vote_count=3,
    )
    assert result.status == "miss"
    assert result.fault_id == ""


def test_semantic_map_no_quorum_is_llm_failed() -> None:
    result = map_unresolved_fault_text(
        llm=QueueLLM(
            [
                "not-json",
                {"fault_id": "F999-madeUp", "reason": "hallucinated"},
                {"fault_id": "F010-slowSQL", "reason": "only valid vote"},
            ]
        ),
        raw_text="slow query",
        reference="- F010-slowSQL: slow SQL",
        dictionary=_dictionary(),
        vote_count=3,
    )
    assert result.status == "llm_failed"
    assert result.fault_id == ""
    assert len(result.votes) == 3


def test_semantic_map_vote_accepts_json_fence() -> None:
    vote = parse_semantic_map_vote(
        '```json\n{"fault_id":"F010-slowSQL","reason":"match"}\n```'
    )
    assert vote.fault_id == "F010-slowSQL"
