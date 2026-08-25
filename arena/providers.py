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
import urllib.error
import urllib.request
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """A request to a model provider failed, either immediately (non-retryable, e.g.
    bad auth, malformed response) or after exhausting retries (transient, e.g. rate
    limit, timeout). Callers can catch this to record a failed task result instead of
    letting one bad request take down an entire suite run."""


# HTTP statuses worth retrying: rate limiting and transient server-side failures.
# Anything else (400 bad request, 401/403 auth, 404 unknown model) is a configuration
# problem that a retry won't fix, so it's raised immediately instead.
_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


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

    def complete_conversation(
        self, spec: ModelSpec, turns: list[str]
    ) -> tuple[str, float, list[dict]]:
        """Run a multi-turn conversation, one user turn at a time, feeding each prior
        assistant reply back in as history (the standard behavior for a stateless
        chat-completions API, which resends the full transcript on every round).

        Returns (final_response_text, total_latency_seconds, transcript), where
        transcript is the full list of {"role", "content"} messages exchanged — callers
        use it to compute cumulative token/cost accounting that reflects what a real
        multi-round exchange actually sends over the wire, not just the final turn.
        """
        raise NotImplementedError


class OpenAICompatProvider(Provider):
    """Talks to any /v1/chat/completions endpoint using only the standard library.

    Retries transient failures (rate limits, transient 5xxs, connection/timeout
    errors) with exponential backoff before giving up. Non-retryable failures (bad
    auth, malformed responses) raise `ProviderError` immediately.
    """

    def __init__(self, max_retries: int = 3, backoff_base_s: float = 0.5, sleep_fn=time.sleep):
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep_fn

    def _post(self, spec: ModelSpec, messages: list[dict]) -> tuple[str, float]:
        api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        payload = json.dumps(
            {
                "model": spec.model,
                "messages": messages,
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

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=spec.timeout_s) as resp:
                    raw = resp.read()
                latency = time.monotonic() - start
            except urllib.error.HTTPError as e:
                if e.code not in _RETRYABLE_HTTP_STATUS:
                    raise ProviderError(
                        f"{spec.name}: non-retryable HTTP {e.code} from provider ({e.reason})"
                    ) from e
                last_err = e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
            else:
                try:
                    body = json.loads(raw)
                    text = body["choices"][0]["message"]["content"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                    raise ProviderError(
                        f"{spec.name}: malformed response (expected choices[0].message.content): {e}"
                    ) from e
                return text, latency

            if attempt < self.max_retries:
                self._sleep(self.backoff_base_s * (2**attempt))

        raise ProviderError(
            f"{spec.name}: request failed after {self.max_retries + 1} attempts: {last_err}"
        ) from last_err

    def complete(self, spec: ModelSpec, prompt: str) -> tuple[str, float]:
        return self._post(spec, [{"role": "user", "content": prompt}])

    def complete_conversation(
        self, spec: ModelSpec, turns: list[str]
    ) -> tuple[str, float, list[dict]]:
        messages: list[dict] = []
        total_latency = 0.0
        final_text = ""
        for turn in turns:
            messages.append({"role": "user", "content": turn})
            text, latency = self._post(spec, messages)
            total_latency += latency
            messages.append({"role": "assistant", "content": text})
            final_text = text
        return final_text, total_latency, messages


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

    def complete_conversation(
        self, spec: ModelSpec, turns: list[str]
    ) -> tuple[str, float, list[dict]]:
        strong = spec.model.endswith("a")
        messages: list[dict] = []
        total_latency = 0.0
        final_text = ""
        for i, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn})
            digest = hashlib.sha256(f"{spec.model}:{'|'.join(turns[: i + 1])}".encode()).hexdigest()
            fake_latency = (int(digest[:4], 16) % 200) / 100.0
            if i == len(turns) - 1:
                text = _mock_multi_turn_answer(turns, strong) or f"mock-response-{digest[:8]}"
            else:
                # Earlier turns just need a plausible acknowledgment; only the final
                # turn is scored, but a real conversation still gets a reply each round.
                text = f"ack-{i + 1}"
            total_latency += fake_latency
            messages.append({"role": "assistant", "content": text})
            final_text = text
        return final_text, total_latency, messages


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


def _instruction_following_answer(text: str, strong: bool) -> str | None:
    """Simulate resistance to in-context distraction: a fixed user instruction competes
    against a second, "louder" instruction embedded in reference material the model is
    quoting from or summarizing (a review note, a system-style banner, a formatting
    request — including a bare "ignore all previous instructions" injection). A model
    with a solid instruction hierarchy still follows the original instruction; a weaker
    one gets pulled into the embedded one, which is the same failure mode that makes
    prompt injection dangerous in agentic/tool-use settings.
    """
    if "zone 4 held at 42 psi" in text:
        if strong:
            return "yes"
        return (
            "the coolant loop pressure remained within the tolerance band for the "
            "full shift, so it was within tolerance."
        )
    if "64 kw circuit budget" in text:
        return "8" if strong else "64 / 8 = 8, so the circuit can support 8 racks."
    if "approved backup power sources" in text:
        if strong:
            return "utility grid, diesel generator, battery ups"
        return "battery ups, utility grid, diesel generator, flywheel storage, fuel cell"
    if "ignore all previous instructions" in text:
        return "yes" if strong else "override"
    if "above the 2.0 db alarm threshold" in text:
        if strong:
            return '{"above_threshold": true}'
        return "| above_threshold |\n| --- |\n| true |"
    return None


def _mock_multi_turn_answer(turns: list[str], strong: bool) -> str | None:
    """Task-aware mock answers for tasks that require carrying state across turns.

    The strong mock reads the full transcript; the weak mock only effectively tracks
    the latest turn, reproducing three real long-conversation failure modes: forgetting
    an early naming/formatting instruction, anchoring on the first-mentioned fact
    instead of a later correction, and losing a numeric constraint stated earlier.
    """
    full_text = " ".join(turns).lower()
    latest = turns[-1].lower()

    if "call the primary cooling loop" in full_text and "tripped its high-pressure" in latest:
        return "loop a" if strong else "insufficient information to determine which loop is available"
    if "circuit budget for this row" in full_text and "how many racks" in latest:
        return "8" if strong else "10"
    if "on-call engineer for this incident is" in full_text and "who is the on-call engineer" in latest:
        return "marcus" if strong else "priya"
    if (
        "answer every question in this conversation with exactly one word" in full_text
        and "is row 12 now on redundant power" in latest
    ):
        return (
            "yes"
            if strong
            else "yes, row 12 is back on redundant power following the completion of scheduled maintenance."
        )
    return None


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
    instruction_following = _instruction_following_answer(text, strong)
    if instruction_following is not None:
        return instruction_following
    if "grid interconnect is capped at 200 mw" in text:
        return "3" if strong else "4"
    if "percentage improvement" in text:
        return "12%" if strong else "11%"
    if "three resources most commonly cited" in text:
        return "power, land, chips" if strong else "power, chips"
    if "quarters memory footprint" in text:
        return "35" if strong else "70"
    if "latest paper on sparse autoencoders" in text:
        return "1. search_arxiv(query)\n2. summarize(text)" if strong else "use search"
    if "sparse autoencoders" in text:
        return (
            "Sparse autoencoders separate superposition into interpretable features."
            if strong
            else "They compress activations."
        )
    if "json key you would read" in text:
        return "latency_s"
    if "read_file(path)" in text and "write_file(path, content)" in text:
        return (
            "Call read_file(path), preserve the existing content, then write_file(path, content)."
            if strong
            else "Read it, then save it."
        )
    if "retry, abort, or escalate" in text:
        # A single transient timeout should be retried, not abandoned -- a weaker
        # heuristic gives up immediately instead. (The retry logic this repo's own
        # OpenAICompatProvider uses is the "strong" behavior here.)
        return "retry" if strong else "abort"
    if "sum the first n elements" in text:
        return "for i in range(n):" if strong else "for i in range(n - 1):"
    if "rate limiter should block requests" in text:
        return "return count >= limit" if strong else "return count > limit"
    if "raises zerodivisionerror whenever cost_usd is 0" in text:
        return (
            "return score / cost_usd if cost_usd != 0 else None"
            if strong
            else "return score / cost_usd"
        )
    if "mutable default argument pitfall" in text:
        return "def add_tag(tag, tags=None):" if strong else "def add_tag(tag, tags=[]):"
    if "returns the original unnormalized" in text:
        return "return result" if strong else "return scores"

    return f"mock-response-{digest[:8]}"


def get_provider(kind: str) -> Provider:
    if kind == "openai_compat":
        return OpenAICompatProvider()
    if kind == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider kind {kind!r}; expected 'openai_compat' or 'mock'")
