"""rcaeval.llm.dashscope — minimal DashScope (Aliyun model service) client.

Default model: qwen3.7-plus (matches the RCA200 Brise judge configuration).
Env vars:
  - DASHSCOPE_API_KEY (required)
  - DASHSCOPE_MAX_TOKENS (optional positive integer override)
  - DASHSCOPE_REASONING_EFFORT (optional provider-supported effort level)

Uses the OpenAI-compatible HTTP endpoint so we don't need the dashscope SDK.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import LLMClient, LLMMessage


def _supports_disable_thinking(model: str) -> bool:
    """Return whether the provider accepts ``enable_thinking=false``.

    Kimi K3 currently rejects the flag with HTTP 400 even though the error
    text mentions temperature.  Keep thinking enabled for that model and let
    the scorer's schema-correction retries handle protocol drift.
    """
    normalized = model.strip().lower()
    return normalized not in {"kimi/kimi-k3", "kimi-k3"}


class DashScopeLLM(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "qwen3.7-plus",
        timeout_s: int | None = None,
        endpoint: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self._api_key:
            raise RuntimeError(
                "DashScopeLLM needs DASHSCOPE_API_KEY env or api_key kwarg."
            )
        self._model = default_model
        if timeout_s is None:
            try:
                timeout_s = int(os.environ.get("DASHSCOPE_TIMEOUT_S", "180"))
            except ValueError:
                timeout_s = 180
        self._timeout = timeout_s
        self._endpoint = endpoint or os.getenv("DASHSCOPE_ENDPOINT")
        if not self._endpoint:
            raise ValueError(
                "DashScope endpoint is required via constructor or DASHSCOPE_ENDPOINT"
            )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "text",
    ) -> str:
        return self._chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            force_disable_thinking=False,
        )

    def chat_structured_retry(
        self,
        messages: list[LLMMessage],
        *,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: str = "json",
    ) -> str:
        """Retry a failed structured call with model-compatible reasoning settings.

        The first judge call still uses the configured model parameters.  A
        caller invokes this only after transport or strict-schema failure.
        Models that accept ``enable_thinking=false`` disable hidden reasoning;
        Kimi K3 keeps thinking enabled because the provider rejects that flag.
        """
        return self._chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            force_disable_thinking=True,
        )

    def _chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: str,
        force_disable_thinking: bool,
    ) -> str:
        configured_max_tokens = os.environ.get("DASHSCOPE_MAX_TOKENS", "").strip()
        if configured_max_tokens:
            try:
                max_tokens = int(configured_max_tokens)
            except ValueError as exc:
                raise ValueError("DASHSCOPE_MAX_TOKENS must be an integer") from exc
            if max_tokens <= 0:
                raise ValueError("DASHSCOPE_MAX_TOKENS must be positive")
        payload = {
            "model": model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        reasoning_effort = os.environ.get("DASHSCOPE_REASONING_EFFORT", "").strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        # qwen3.6-plus and other reasoning Qwen variants emit hidden
        # reasoning_content by default on DashScope, which inflates token
        # cost and confuses callers that only want the final JSON. Top-level
        # enable_thinking=false is the documented DashScope knob.
        selected_model = str(payload["model"])
        if _supports_disable_thinking(selected_model) and (
            force_disable_thinking
            or os.environ.get("DASHSCOPE_DISABLE_THINKING", "1") != "0"
        ):
            payload["enable_thinking"] = False

        body = self._request_with_retry(payload)
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = (
            message.get("reasoning_content") if isinstance(message, dict) else None
        )

        # Some reasoning models can spend the entire completion budget on
        # hidden reasoning and return no JSON content at all. Repeating the
        # same request with the same settings is ineffective. For structured
        # judge calls only, retry once with thinking disabled when supported.
        # Kimi K3 rejects that flag, so retain thinking and expand only the
        # fallback completion budget. The initial judge request remains on
        # the caller's configured max_tokens.
        if (
            response_format == "json"
            and isinstance(first, dict)
            and first.get("finish_reason") == "length"
            and (not isinstance(content, str) or not content.strip())
            and isinstance(reasoning, str)
            and reasoning.strip()
            and payload.get("enable_thinking") is not False
        ):
            fallback_payload = dict(payload)
            if _supports_disable_thinking(selected_model):
                fallback_payload["enable_thinking"] = False
            else:
                fallback_max_tokens = payload.get("max_tokens")
                if not isinstance(fallback_max_tokens, int):
                    raise TypeError("DashScope max_tokens must be an integer")
                fallback_payload["max_tokens"] = max(
                    fallback_max_tokens * 2,
                    8192,
                )
            body = self._request_with_retry(fallback_payload)

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("DashScope response has no choices")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise TypeError("DashScope response has no message")
        content = first["message"].get("content")
        if not isinstance(content, str):
            raise TypeError("DashScope response content is not a string")
        return content

    def _request_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        """Retry rate limits and transient transport failures, then fail closed."""
        last_error: Exception | None = None
        for attempt in range(4):
            request = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = json.loads(response.read())
                if not isinstance(body, dict):
                    raise TypeError("DashScope response is not a JSON object")
                return body
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"DashScope HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
        error_type = type(last_error).__name__ if last_error is not None else "unknown"
        raise RuntimeError(
            f"DashScope request failed after 4 attempts; last_error_type={error_type}"
        ) from last_error
