"""Tests that a provider failure on one task degrades gracefully instead of taking
down the whole suite run — the failure mode `arena.providers.OpenAICompatProvider`
raises `ProviderError` for after exhausting retries."""
import json

import pytest
import yaml

from arena.providers import ModelSpec, ProviderError
from arena.report import build_report
from arena.runner import load_model_specs, run_suite
from arena.tasks import load_tasks


class _FlakyProvider:
    """Fails on the 2nd call, succeeds on every other one — simulates a single
    transient provider outage partway through a suite run."""

    def __init__(self):
        self.calls = 0

    def complete(self, spec, prompt):
        self.calls += 1
        if self.calls == 2:
            raise ProviderError("simulated: rate limit exceeded after retries")
        return "mock-response", 0.01

    def complete_conversation(self, spec, turns):
        raise ProviderError("simulated: multi-turn not exercised in this fake")


def test_run_suite_records_provider_error_without_crashing(tmp_path, monkeypatch):
    tasks = [t for t in load_tasks("tasks") if not t.is_multi_turn][:4]
    monkeypatch.setattr("arena.runner.get_provider", lambda kind: _FlakyProvider())
    spec = ModelSpec(name="flaky", base_url="", model="flaky")
    out_dir = tmp_path / "results"

    run_suite(tasks, [("mock", spec)], out_dir)

    records = [json.loads(line) for line in (out_dir / "flaky.jsonl").read_text().splitlines()]
    assert len(records) == len(tasks)  # the suite kept going past the failed task

    errored = [r for r in records if r["error"]]
    assert len(errored) == 1
    assert "rate limit" in errored[0]["error"]
    assert errored[0]["score"] == 0.0
    assert errored[0]["output"] == ""

    ok = [r for r in records if not r["error"]]
    assert len(ok) == len(tasks) - 1
    for r in ok:
        assert r["output"] == "mock-response"


def test_report_isolates_provider_errors_from_genuine_scores(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    records = [
        {"task_id": "t-1", "category": "reasoning", "model": "flaky", "output": "x",
         "score": 1.0, "latency_s": 0.1, "input_tokens_est": 1, "output_tokens_est": 1,
         "estimated_cost_usd": 0.0, "input_cost_per_million": 0.0, "output_cost_per_million": 0.0,
         "error": None},
        {"task_id": "t-2", "category": "reasoning", "model": "flaky", "output": "",
         "score": 0.0, "latency_s": 0.0, "input_tokens_est": 0, "output_tokens_est": 0,
         "estimated_cost_usd": 0.0, "input_cost_per_million": 0.0, "output_cost_per_million": 0.0,
         "error": "simulated: rate limit exceeded after retries"},
    ]
    (out_dir / "flaky.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    report = build_report(out_dir)

    assert "errors" in report  # column header
    assert "| 1 | flaky | 0.50 |" in report  # one genuine 1.0 and one errored 0.0 averages to 0.50
    assert "task run(s) failed at the provider level" in report


def test_load_model_specs_rejects_duplicate_model_names(tmp_path):
    """Two config entries sharing a display name would silently overwrite each
    other's `{name}.jsonl` result file in `run_suite` -- one model's entire
    evaluation would vanish from the report with no error. This should be
    rejected up front, the same way `load_tasks` rejects duplicate task ids."""
    config_path = tmp_path / "dupe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"name": "glm-5.3", "provider": "mock", "model": "glm-5.3-a"},
                    {"name": "glm-5.3", "provider": "mock", "model": "glm-5.3-b"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate model names"):
        load_model_specs(config_path)


def test_load_model_specs_allows_distinct_names_with_shared_provider_model_id(tmp_path):
    """Two distinct display names are fine even if they happen to point at the same
    underlying provider-side model id (e.g. comparing temperature/prompt variants)."""
    config_path = tmp_path / "ok.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"name": "glm-5.3-run-a", "provider": "mock", "model": "glm-5.3"},
                    {"name": "glm-5.3-run-b", "provider": "mock", "model": "glm-5.3"},
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_model_specs(config_path)

    assert [spec.name for _, spec in specs] == ["glm-5.3-run-a", "glm-5.3-run-b"]


def test_report_shows_no_error_note_when_results_have_no_errors(tmp_path):
    out_dir = tmp_path / "empty_results"
    out_dir.mkdir()
    record = {
        "task_id": "t-1", "category": "reasoning", "model": "clean", "output": "x",
        "score": 1.0, "latency_s": 0.1, "input_tokens_est": 1, "output_tokens_est": 1,
        "estimated_cost_usd": 0.0, "input_cost_per_million": 0.0, "output_cost_per_million": 0.0,
        "error": None,
    }
    (out_dir / "clean.jsonl").write_text(json.dumps(record) + "\n")

    report = build_report(out_dir)

    assert "task run(s) failed at the provider level" not in report
