"""Command-line entry point: `python -m arena.cli run|report ...`"""
from __future__ import annotations

import argparse
import sys

from .report import build_report
from .runner import load_model_specs, run_suite
from .tasks import load_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a task suite against configured models")
    run_p.add_argument("--config", required=True, help="path to models YAML config")
    run_p.add_argument("--tasks", required=True, help="path to tasks directory")
    run_p.add_argument("--out", required=True, help="output directory for result JSONL files")

    report_p = sub.add_parser("report", help="build a markdown report from results")
    report_p.add_argument("--results", required=True, help="results directory (from `run --out`)")
    report_p.add_argument("--out", required=True, help="output markdown file path")
    report_p.add_argument(
        "--chart",
        help="optional output path for a PNG bar chart of overall score per model "
        "(requires matplotlib)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        tasks = load_tasks(args.tasks)
        specs = load_model_specs(args.config)
        if not tasks:
            print(f"no tasks found in {args.tasks}", file=sys.stderr)
            return 1
        run_suite(tasks, specs, args.out)
        return 0

    if args.command == "report":
        report = build_report(args.results)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        if args.chart:
            from .chart import build_chart

            build_chart(args.results, args.chart)
            print(f"wrote chart -> {args.chart}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
