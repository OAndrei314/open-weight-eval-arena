"""Aggregates per-model JSONL result files into a markdown comparison report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def load_results(results_dir: str | Path) -> dict[str, list[dict]]:
    """Load every `*.jsonl` result file in a directory, keyed by model name.

    Shared by `build_report` and `arena.chart.build_chart` so both read the exact
    same on-disk format from a single place.
    """
    results_dir = Path(results_dir)
    by_model: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            by_model[records[0]["model"]] = records
    return by_model


def build_report(results_dir: str | Path) -> str:
    by_model = load_results(results_dir)
    if not by_model:
        return "# Arena Report\n\nNo results found.\n"

    categories = sorted({r["category"] for recs in by_model.values() for r in recs})

    lines = ["# Arena Report", ""]
    header = [
        "model",
        "overall",
        *categories,
        "avg_latency_s",
        "cost_per_1k_tasks_usd",
        "score_per_usd",
        "errors",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for model, recs in sorted(by_model.items()):
        overall = sum(r["score"] for r in recs) / len(recs)
        avg_latency = sum(r["latency_s"] for r in recs) / len(recs)
        total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in recs)
        cost_per_1k = total_cost / len(recs) * 1000 if total_cost > 0 else None
        score_per_usd = sum(r["score"] for r in recs) / total_cost if total_cost > 0 else None
        by_cat = defaultdict(list)
        for r in recs:
            by_cat[r["category"]].append(r["score"])
        cat_scores = [
            f"{(sum(by_cat[c]) / len(by_cat[c])):.2f}" if by_cat[c] else "-" for c in categories
        ]
        error_count = sum(1 for r in recs if r.get("error"))
        row = [
            model,
            f"{overall:.2f}",
            *cat_scores,
            f"{avg_latency:.3f}",
            _fmt_optional(cost_per_1k, 4),
            _fmt_optional(score_per_usd, 1),
            str(error_count) if error_count else "-",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "Cost columns use configured $/million-token rates and deterministic token estimates "
        "when provider usage data is unavailable."
    )
    total_errors = sum(1 for recs in by_model.values() for r in recs if r.get("error"))
    if total_errors:
        lines.append("")
        lines.append(
            f"**{total_errors} task run(s) failed at the provider level** (rate limits, "
            "timeouts, malformed responses) after exhausting retries and are scored 0 in "
            "`overall` — the `errors` column isolates these from genuine wrong answers so "
            "a low score isn't misread as a model quality result when it was actually an "
            "infra failure."
        )
    lines.append("")
    lines.append(f"_{sum(len(r) for r in by_model.values())} task runs across "
                  f"{len(by_model)} model(s), {len(categories)} categories._")
    return "\n".join(lines) + "\n"
