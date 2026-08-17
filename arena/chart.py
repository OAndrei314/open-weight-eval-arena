"""Renders the score comparison from `results/*.jsonl` as a PNG bar chart.

Kept separate from `report.py` so the core JSONL -> markdown path (used by every
test and the mock-provider quickstart) never depends on matplotlib — only
`arena report --chart` does, via a lazy import in `cli.py`.
"""
from __future__ import annotations

from pathlib import Path

from .report import load_results


def build_chart(results_dir: str | Path, out_path: str | Path) -> None:
    """Write a horizontal bar chart of overall score per model to `out_path`.

    Raises ValueError if `results_dir` has no result files, matching
    `build_report`'s "No results found" case rather than silently emitting an
    empty image.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display, safe for CI and scripts
    import matplotlib.pyplot as plt

    by_model = load_results(results_dir)
    if not by_model:
        raise ValueError(f"no results found in {results_dir}")

    models = sorted(by_model, key=lambda m: sum(r["score"] for r in by_model[m]) / len(by_model[m]))
    overall_scores = [sum(r["score"] for r in by_model[m]) / len(by_model[m]) for m in models]

    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.6 * len(models))))
    ax.barh(models, overall_scores, color="#4C72B0")
    ax.set_xlabel("overall score")
    ax.set_xlim(0, 1)
    ax.set_title("Arena: overall score by model")
    for i, v in enumerate(overall_scores):
        ax.text(min(v + 0.02, 0.97), i, f"{v:.2f}", va="center")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
