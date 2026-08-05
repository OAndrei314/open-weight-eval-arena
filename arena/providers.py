"""Model providers.

Real providers talk to any OpenAI-compatible /v1/chat/completions endpoint, which covers
most open-weight model hosts (OpenRouter, Fireworks, Together, Groq, local vLLM / Ollama /
llama.cpp servers with an OpenAI-compatible shim).

MockProvider exists so the whole pipeline is runnable and testable with zero API keys and
zero network access — useful for CI and for anyone cloning this repo before they've wired up
real credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # display name, e.g. "glm-5.2"
    base_url: str
    model: str  # provider-side model id, e.g. "zhipu/glm-5.2"
    api_key_env: str | None = None
    timeout_s: float = 60.0


class Provider:
    def complete(self, spec: ModelSpec, prompt: str) -> tuple[str, float]:
        """Return (completion_text, latency_seconds)."""
        raise NotImplementedError


class OpenAICompatProvider(Provider):
    """Talks to any /v1/chat/completions endpoint using only the standard library."""

    def complete(self, spec: ModelSpec, prompt: str) -> tuple[str, float]:
        api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        payload = json.dumps(
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url=f"{spec.base_url.rstrip('/')}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=spec.timeout_s) as resp:
            body = json.loads(resp.read())
        latency = time.monotonic() - start
        text = body["choices"][0]["message"]["content"]
        return text, latency


class MockProvider(Provider):
    """Deterministic stand-in: hashes (model, prompt) into a stable pseudo-completion.

    This intentionally does NOT try to be a good model — it exists so `arena.runner` has
    something real to exercise end-to-end without network access, for tests and for a
    zero-setup first run.
    """

    def complete(self, spec: ModelSpec, prompt: str) -> tuple[str, float]:
        digest = hashlib.sha256(f"{spec.model}:{prompt}".encode()).hexdigest()
        # Fake but stable "latency" so report formatting/sorting has something to show.
        fake_latency = (int(digest[:4], 16) % 200) / 100.0
        return f"mock-response-{digest[:8]}", fake_latency


def get_provider(kind: str) -> Provider:
    if kind == "openai_compat":
        return OpenAICompatProvider()
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider kind {kind!r}; expected 'openai_compat' or 'mock'")
