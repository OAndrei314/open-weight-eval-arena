"""Tests for the leaderboard ranking in `build_report`."""
import json

from arena.report import build_report


def _write_model(out_dir, model: str, scores: list[float]) -> None:
    records = [
        {
            "task_id": f"{model}-{i}",
            "category": "reasoning",
            "model": model,
            "output": "x",
            "score": s,
            "latency_s": 0.1,
            "input_tokens_est": 1,
            "output_tokens_est": 1,
            "estimated_cost_usd": 0.0,
            "input_cost_per_million": 0.0,
            "output_cost_per_million": 0.0,
            "error": None,
        }
        for i, s in enumerate(scores)
    ]
    (out_dir / f"{model}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _rank_order(report: str) -> list[str]:
    """Extract model names in the order they appear in the report table, by rank."""
    rows = [
        line.split("|")[2].strip()
        for line in report.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    return rows


def test_build_report_ranks_by_overall_score_descending(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    # Written in an order that does NOT match score order or alphabetical order,
    # so a passing test can't be explained by either sort accidentally lining up.
    _write_model(out_dir, "mid", [0.5, 0.5])
    _write_model(out_dir, "best", [1.0, 1.0])
    _write_model(out_dir, "worst", [0.0, 0.0])

    report = build_report(out_dir)

    assert _rank_order(report) == ["best", "mid", "worst"]
    assert "| 1 | best | 1.00 |" in report
    assert "| 2 | mid | 0.50 |" in report
    assert "| 3 | worst | 0.00 |" in report


def test_build_report_breaks_ties_by_model_name(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    _write_model(out_dir, "zeta", [0.5])
    _write_model(out_dir, "alpha", [0.5])

    report = build_report(out_dir)

    # Equal scores: deterministic tie-break is alphabetical by model name, not
    # insertion order or a coincidence of alphabetical == score order.
    assert _rank_order(report) == ["alpha", "zeta"]
