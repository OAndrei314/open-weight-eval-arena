"""Scorers. Deliberately simple and inspectable — no LLM-judges-LLM circularity.

Each scorer takes (model_output, reference) and returns a float in [0, 1].
"""
from __future__ import annotations

import re

ScoreFn = "callable[[str, str], float]"


def exact_match(output: str, reference: str) -> float:
    return 1.0 if output.strip().lower() == reference.strip().lower() else 0.0


def keyword_coverage(output: str, reference: str) -> float:
    """Reference is a comma-separated list of required keywords; score = fraction present."""
    keywords = [k.strip().lower() for k in reference.split(",") if k.strip()]
    if not keywords:
        return 0.0
    out_lower = output.lower()
    hits = sum(1 for k in keywords if k in out_lower)
    return hits / len(keywords)


def regex_rubric(output: str, reference: str) -> float:
    """Reference is a regex; score = 1.0 if it matches anywhere in the output (re.DOTALL)."""
    try:
        return 1.0 if re.search(reference, output, re.DOTALL) else 0.0
    except re.error as e:
        raise ValueError(f"invalid regex reference {reference!r}: {e}") from e


SCORERS = {
    "exact_match": exact_match,
    "keyword_coverage": keyword_coverage,
    "regex_rubric": regex_rubric,
}


def score(scorer_name: str, output: str, reference: str) -> float:
    try:
        fn = SCORERS[scorer_name]
    except KeyError:
        raise ValueError(
            f"unknown scorer {scorer_name!r}; available: {sorted(SCORERS)}"
        ) from None
    return fn(output, reference)
