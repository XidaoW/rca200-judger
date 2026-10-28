"""rcaeval.llm.base — LLM client interface.

rcaeval's scorer makes 4 kinds of LLM calls (depending on which milestone is wired):

  M1  RCA_PROCESS_REVIEW_PROMPT  → Process 三维 review JSON
  M2  RCA_LLM_CAUSAL_JUDGE       → F LLM rescue when exact/cluster miss
  M4  FAULT_TYPE_EXTRACTION_*    → trace text → fault_id candidates
  M4  ENTITY_EXTRACTION_*        → trace text → entity_name candidates

Each call site declares the response schema it expects; the LLMClient implementation
is responsible for "best-effort" JSON-only output. We do not depend on any
provider's structured-output feature — caller does its own JSON parse + retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """Minimal chat interface. Concrete impls: stub, dashscope, openai, anthropic, vllm."""

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "text",  # "text" | "json"
    ) -> str:
        """Return the assistant message content as a single string.

        Caller handles JSON-mode tolerance: even when response_format="json",
        impls may return Markdown-fenced JSON; rcaeval.process strips that fence
        before json.loads.
        """
        ...
