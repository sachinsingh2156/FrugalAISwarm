"""
Ollama HTTP client.
Wraps the /api/generate endpoint and exposes:
  - generate()       : text completion with optional logprobs
  - chat()           : chat-style completion
  - mean_log_prob()  : uncertainty signal from token log-probabilities
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import httpx

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT, get_model


class OllamaClient:
    """Thin synchronous client for a locally-hosted Ollama endpoint."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str | None = None,
        timeout: int = OLLAMA_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or get_model()
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ── Low-level generate ────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: bool = False,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Call /api/generate (non-streaming).
        Returns the full Ollama response dict, augmented with:
          - 'text'          : the generated text
          - 'tokens_used'   : prompt_eval_count + eval_count
          - 'mean_log_prob' : mean token log-prob (if logprobs=True and available)

        model_override: if set, uses this model instead of self.model (A2/A4).
        """
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if logprobs:
            payload["options"]["logprobs"] = True

        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        data["latency_s"] = time.perf_counter() - t0
        data["text"] = data.get("response", "")
        data["tokens_used"] = (
            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        )

        # Compute mean log-prob from token probabilities if present
        if logprobs and "logprobs" in data:
            lps = data["logprobs"].get("token_logprobs", [])
            data["mean_log_prob"] = float(
                sum(lp for lp in lps if lp is not None) / len(lps)
            ) if lps else 0.0
        else:
            data["mean_log_prob"] = None

        return data

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        """
        Call /api/chat (non-streaming).
        messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if logprobs:
            payload["options"]["logprobs"] = True

        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        data["latency_s"] = time.perf_counter() - t0
        msg = data.get("message", {})
        data["text"] = msg.get("content", "")
        data["tokens_used"] = (
            data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        )
        return data

    # ── Utility ───────────────────────────────────────────────────────────────

    def mean_log_prob(self, response_data: dict[str, Any]) -> float | None:
        """
        Extract the mean token log-probability from a generate() response.
        Returns None if logprob data is unavailable (Ollama version dependent).
        """
        return response_data.get("mean_log_prob")

    def list_models(self) -> list[str]:
        """Return model tags available in this Ollama instance."""
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def is_available(self) -> bool:
        """Ping the Ollama server."""
        try:
            self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
