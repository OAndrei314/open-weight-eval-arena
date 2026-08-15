"""Simple token and cost estimates for model-evaluation runs."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token proxy used when provider usage data is unavailable."""
    pieces = [part for part in text.replace("\n", " ").split(" ") if part]
    return max(1, int(round(len(pieces) * 1.3)))


def estimate_conversation_tokens(transcript: list[dict]) -> tuple[int, int]:
    """Cumulative (input_tokens, output_tokens) for a multi-turn exchange, assuming a
    stateless chat API that resends the full prior transcript on every round — the
    standard behavior for OpenAI-compatible /v1/chat/completions endpoints. A naive
    per-message sum would understate input cost, since the same early turns get
    re-sent (and re-billed) on every later round.
    """
    input_tokens = 0
    output_tokens = 0
    history: list[dict] = []
    for msg in transcript:
        history.append(msg)
        if msg["role"] == "user":
            input_tokens += sum(estimate_tokens(m["content"]) for m in history)
        else:
            output_tokens += estimate_tokens(msg["content"])
    return input_tokens, output_tokens


def estimate_request_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    input_cost = input_tokens / 1_000_000 * input_cost_per_million
    output_cost = output_tokens / 1_000_000 * output_cost_per_million
    return input_cost + output_cost
