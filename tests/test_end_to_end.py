"""End-to-end test using only the MockProvider — no network access required."""
import json

from arena.report import build_report
from arena.runner import load_model_specs, run_suite
from arena.tasks import load_tasks


def test_full_pipeline_with_mock_provider(tmp_path):
    tasks = load_tasks("tasks")
    specs = load_model_specs("configs/mock.yaml")
    out_dir = tmp_path / "results"

    run_suite(tasks, specs, out_dir)

    result_files = list(out_dir.glob("*.jsonl"))
    assert len(result_files) == len(specs)
    first_record = json.loads(result_files[0].read_text().splitlines()[0])
    assert first_record["input_tokens_est"] > 0
    assert first_record["output_tokens_est"] > 0
    assert first_record["estimated_cost_usd"] > 0

    report = build_report(out_dir)
    assert "# Arena Report" in report
    assert "mock-a" in report
    assert "mock-b" in report
    assert "long_context_recall" in report
    assert "code_repair" in report
    assert "cost_per_1k_tasks_usd" in report
    assert "score_per_usd" in report
