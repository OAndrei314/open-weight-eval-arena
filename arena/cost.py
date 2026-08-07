"""Simple token and cost estimates for model-evaluation runs."""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token proxy used when provider usage data is unavailable."""
    pieces = [part for part in text.replace("\n", " ").split(" ") if part]
    return max(1, int(round(len(pieces) * 1.3)))


def estimate_request_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    input_cost = input_tokens / 1_000_000 * input_cost_per_million
    output_cost = output_tokens / 1_000_000 * output_cost_per_million
    return input_cost + output_cost
