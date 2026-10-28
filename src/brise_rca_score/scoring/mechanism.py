"""Standalone Brise-compatible mechanism consistency gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, LLMMessage
from .models import FaultTypeScoreEvidence, FaultTypeTier

MECHANISM_GATE_PROMPT_VERSION = "2026-08-13.2"


def load_mechanism_gate_prompt() -> str:
    """Load the frozen Brise active mechanism-gate prompt snapshot."""
    return (
        files("brise_rca_score.data.prompts") / "RCA_MECHANISM_GATE_PROMPT.md"
    ).read_text(encoding="utf-8")


def mechanism_gate_prompt_sha256() -> str:
    return hashlib.sha256(load_mechanism_gate_prompt().encode("utf-8")).hexdigest()


MechanismVerdictName = Literal[
    "consistent", "label_only", "contradicted", "insufficient_gt"
]


class MechanismVote(BaseModel):
    verdict: MechanismVerdictName
    agent_mechanism_claim: str = ""
    reason: str = ""


class MechanismOutcome(BaseModel):
    status: str
    reason: str = ""
    claim: str = ""
    votes: list[str] = Field(default_factory=list)
    capped_score: float | None = None


def _parse_vote(raw: str) -> MechanismVote:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return MechanismVote.model_validate(json.loads(text.strip()))


def evaluate_mechanism(
    *,
    llm: LLMClient,
    expected: str,
    expected_desc: str,
    gt_mechanism: str,
    actual_list: list[str],
    agent_conclusion: str,
    agent_chain_text: str,
    model: str,
    temperature: float,
    vote_count: int = 3,
) -> MechanismOutcome:
    """Compare an Agent mechanism claim with GT using Brise's majority rule."""
    if not gt_mechanism.strip():
        return MechanismOutcome(status="skipped_no_gt")
    if not any(item.strip() for item in actual_list) and not agent_conclusion.strip():
        return MechanismOutcome(status="skipped_no_agent_claim")
    prompt = load_mechanism_gate_prompt().format(
        expected=expected,
        expected_desc=expected_desc or "未知",
        gt_mechanism=gt_mechanism,
        actual_list="\n".join(f"- {item}" for item in actual_list if item.strip()),
        agent_conclusion=agent_conclusion or "（无结论）",
        agent_chain_text=agent_chain_text or "（无 Agent 传播链）",
    )
    votes: list[MechanismVote] = []
    for _ in range(max(1, vote_count)):
        try:
            raw = llm.chat(
                [LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=temperature,
                response_format="json",
            )
            votes.append(_parse_vote(raw))
        except (ValueError, json.JSONDecodeError):
            continue
        if len(votes) >= 2 and votes[-1].verdict == votes[-2].verdict:
            break
    if not votes:
        raise RuntimeError("mechanism judge returned no valid votes")
    counts = Counter(vote.verdict for vote in votes)
    severity = {
        "consistent": 0,
        "insufficient_gt": 0,
        "label_only": 1,
        "contradicted": 2,
    }
    highest = max(counts.values())
    winner_name = max(
        (name for name, count in counts.items() if count == highest),
        key=lambda name: severity[name],
    )
    winner = next(vote for vote in votes if vote.verdict == winner_name)
    cap = None
    if winner.verdict == "label_only":
        cap = FaultTypeTier.CLUSTER.default_score
    elif winner.verdict == "contradicted":
        cap = min(
            FaultTypeTier.INJECTION_POINT.default_score,
            FaultTypeTier.CLUSTER.default_score,
        )
    return MechanismOutcome(
        status=winner.verdict,
        reason=winner.reason,
        claim=winner.agent_mechanism_claim,
        votes=[vote.verdict for vote in votes],
        capped_score=cap,
    )


def apply_mechanism_outcome(
    evidence: FaultTypeScoreEvidence,
    outcome: MechanismOutcome,
) -> FaultTypeScoreEvidence:
    """Apply a gate outcome monotonically: it may cap a score, never raise it."""
    evidence.mechanism_gate_status = outcome.status
    evidence.mechanism_gate_reason = outcome.reason
    evidence.mechanism_gate_claim = outcome.claim
    evidence.mechanism_gate_votes = list(outcome.votes)
    if outcome.capped_score is None or evidence.score <= outcome.capped_score:
        return evidence
    evidence.mechanism_gate_prior_score = evidence.score
    evidence.mechanism_gate_prior_tier = evidence.match_tier
    evidence.score = round(outcome.capped_score, 4)
    evidence.match_tier = f"mechanism_gate_{outcome.status}"
    evidence.scoring_method = "mechanism_gate_capped"
    evidence.match_detail = (
        f"{evidence.match_detail}；机制一致性核对：{outcome.status}——{outcome.reason}"
    )
    return evidence
