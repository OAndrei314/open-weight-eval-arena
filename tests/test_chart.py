"""Chart generation tests — matplotlib runs headless (Agg backend), no network access."""
import pytest

from arena.chart import build_chart
from arena.runner import load_model_specs, run_suite
from arena.tasks import load_tasks


def test_build_chart_writes_a_nonempty_png(tmp_path):
    tasks = load_tasks("tasks")
    specs = load_model_specs("configs/mock.yaml")
    results_dir = tmp_path / "results"
    run_suite(tasks, specs, results_dir)

    chart_path = tmp_path / "chart.png"
    build_chart(results_dir, chart_path)

    assert chart_path.exists()
    data = chart_path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    assert len(data) > 1000


def test_build_chart_raises_on_empty_results_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="no results found"):
        build_chart(empty_dir, tmp_path / "chart.png")


def test_build_chart_creates_missing_parent_directories(tmp_path):
    tasks = load_tasks("tasks")
    specs = load_model_specs("configs/mock.yaml")
    results_dir = tmp_path / "results"
    run_suite(tasks, specs, results_dir)

    nested_path = tmp_path / "nested" / "dir" / "chart.png"
    build_chart(results_dir, nested_path)

    assert nested_path.exists()
