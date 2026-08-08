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
import re
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
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


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
        return _mock_answer(spec.model, prompt, digest), fake_latency


_NEEDLE_RE = re.compile(r"facility access code to record in the log is exactly: ([A-Z0-9-]+)\.")


def _long_context_recall_answer(prompt: str, strong: bool) -> str:
    """Simulate needle-in-haystack recall, including the well-documented
    "lost in the middle" effect: weaker models are reliable when the needle sits
    near the start or end of the context but drop it when it's buried in the
    middle third, even though the strong model still finds it either way.
    """
    match = _NEEDLE_RE.search(prompt)
    if not match:
        return "not found"
    code = match.group(1)
    if strong:
        return code
    needle_fraction = match.start() / len(prompt)
    if needle_fraction < 0.2 or needle_fraction > 0.8:
        return code
    return "not found"


def _mock_answer(model: str, prompt: str, digest: str) -> str:
    """Task-aware deterministic mock answers for the bundled fixture suite.

    `mock-a` represents a stronger, more expensive model. `mock-b` answers a smaller
    subset correctly, so reports show an actual quality/cost tradeoff without a network
    call or API key.
    """
    text = prompt.lower()
    strong = model.endswith("a")

    if "facility access code" in text:
        return _long_context_recall_answer(prompt, strong)
    if "grid interconnect is capped at 200 mw" in text:
        return "3" if strong else "4"
    if "percentage improvement" in text:
        return "12%" if strong else "11%"
    if "three resources most commonly cited" in text:
        return "power, land, chips" if strong else "power, chips"
    if "quarters memory footprint" in text:
        return "35" if strong else "70"
    if "sparse autoencoders" in text:
        return (
            "Sparse autoencoders separate superposition into interpretable features."
            if strong
            else "They compress activations."
        )
    if "latest paper on sparse autoencoders" in text:
        return "1. search_arxiv(query)\n2. summarize(text)" if strong else "use search"
    if "json key you would read" in text:
        return "latency_s"
    if "read_file(path)" in text and "write_file(path, content)" in text:
        return (
            "Call read_file(path), preserve the existing content, then write_file(path, content)."
            if strong
            else "Read it, then save it."
        )
    if "retry, abort, or escalate" in text:
        return "retry"

    return f"mock-response-{digest[:8]}"


def get_provider(kind: str) -> Provider:
    if kind == "openai_compat":
        return OpenAICompatProvider()
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider kind {kind!r}; expected 'openai_compat' or 'mock'")
