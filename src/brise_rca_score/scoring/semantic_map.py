"""Brise-compatible semantic mapping for unresolved fault claims."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.resources import files
from typing import Literal, Mapping

from pydantic import BaseModel, Field

from ..llm.base import LLMClient, LLMMessage

SEMANTIC_MAP_PROMPT_VERSION = "2026-08-04.1"


class SemanticMapVote(BaseModel):
    fault_id: str = ""
    reason: str = ""


class SemanticFaultMapping(BaseModel):
    fault_id: str = ""
    reason: str = ""
    status: Literal["hit", "miss", "llm_failed"] = "miss"
    votes: list[SemanticMapVote | None] = Field(default_factory=list)
    prompt_version: str = SEMANTIC_MAP_PROMPT_VERSION
    prompt_sha256: str = ""


def load_semantic_map_prompt() -> str:
    """Load the frozen Brise active semantic-map prompt snapshot."""
    return (
        files("brise_rca_score.data.prompts") / "RCA_SEMANTIC_MAP_PROMPT.md"
    ).read_text(encoding="utf-8")


def semantic_map_prompt_sha256() -> str:
    return hashlib.sha256(load_semantic_map_prompt().encode("utf-8")).hexdigest()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    stripped = stripped[first_newline + 1 :] if first_newline >= 0 else ""
    stripped = stripped.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def parse_semantic_map_vote(raw_text: str) -> SemanticMapVote:
    """Parse one strict JSON vote; an empty ID is valid only with a reason."""
    try:
        payload = json.loads(_strip_json_fence(raw_text))
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic map response is not a single JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic map response is not a single JSON object")
    vote = SemanticMapVote.model_validate(payload)
    if not vote.fault_id.strip() and not vote.reason.strip():
        raise ValueError("semantic map response is empty vote without reason")
    return vote


def _normalize_fault_id(
    fault_id: str,
    dictionary: Mapping[str, str],
) -> str | None:
    value = fault_id.strip()
    if not value:
        return ""
    key = value.lower()
    direct = dictionary.get(key)
    if direct:
        return direct
    canonical_values = tuple(dict.fromkeys(dictionary.values()))
    for canonical in canonical_values:
        if canonical.lower() == key:
            return canonical
    short_matches = [
        canonical
        for canonical in canonical_values
        if canonical.split("-", 1)[0].lower() == key
    ]
    return short_matches[0] if len(short_matches) == 1 else None


def map_unresolved_fault_text(
    *,
    llm: LLMClient,
    raw_text: str,
    reference: str,
    dictionary: Mapping[str, str],
    context: str = "",
    model: str = "",
    temperature: float = 0.0,
    vote_count: int = 3,
) -> SemanticFaultMapping:
    """Map free-text fault claims using Brise's active prompt and majority vote."""
    text = str(raw_text or "").strip()
    prompt_sha = semantic_map_prompt_sha256()
    if not text or not dictionary:
        return SemanticFaultMapping(prompt_sha256=prompt_sha)
    if vote_count < 1:
        raise ValueError("vote_count must be positive")

    prompt = load_semantic_map_prompt().format(
        reference=reference,
        raw_text=text,
        context=context.strip() or "（无）",
    )
    votes: list[SemanticMapVote | None] = []
    normalized: list[SemanticMapVote] = []
    for _ in range(vote_count):
        try:
            raw = llm.chat(
                [LLMMessage(role="user", content=prompt)],
                model=model,
                temperature=temperature,
                response_format="json",
            )
            vote = parse_semantic_map_vote(raw)
            canonical = _normalize_fault_id(vote.fault_id, dictionary)
            if canonical is None:
                votes.append(None)
                continue
            normalized_vote = vote.model_copy(update={"fault_id": canonical})
            votes.append(normalized_vote)
            normalized.append(normalized_vote)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            votes.append(None)

    quorum = vote_count // 2 + 1
    if len(normalized) < quorum:
        return SemanticFaultMapping(
            status="llm_failed",
            votes=votes,
            prompt_sha256=prompt_sha,
        )
    counts = Counter(vote.fault_id for vote in normalized)
    winner_id, winner_count = max(counts.items(), key=lambda item: item[1])
    if winner_count < quorum:
        return SemanticFaultMapping(
            status="llm_failed",
            votes=votes,
            prompt_sha256=prompt_sha,
        )
    winner = next(vote for vote in normalized if vote.fault_id == winner_id)
    return SemanticFaultMapping(
        fault_id=winner.fault_id,
        reason=winner.reason,
        status="hit" if winner.fault_id else "miss",
        votes=votes,
        prompt_sha256=prompt_sha,
    )
