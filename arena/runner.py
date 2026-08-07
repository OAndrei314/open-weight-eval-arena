"""Runs a task suite against a list of model configs and writes JSONL results."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .cost import estimate_request_cost_usd, estimate_tokens
from .providers import ModelSpec, get_provider
from .scoring import score
from .tasks import Task, load_tasks


def load_model_specs(config_path: str | Path) -> list[tuple[str, ModelSpec]]:
    """Returns list of (provider_kind, ModelSpec)."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    specs = []
    for entry in raw["models"]:
        kind = entry.get("provider", "openai_compat")
        spec = ModelSpec(
            name=entry["name"],
            base_url=entry.get("base_url", ""),
            model=entry.get("model", entry["name"]),
            api_key_env=entry.get("api_key_env"),
            timeout_s=float(entry.get("timeout_s", 60.0)),
            input_cost_per_million=float(entry.get("input_cost_per_million", 0.0)),
            output_cost_per_million=float(entry.get("output_cost_per_million", 0.0)),
        )
        specs.append((kind, spec))
    return specs


def run_suite(
    tasks: list[Task], model_specs: list[tuple[str, ModelSpec]], out_dir: str | Path
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, spec in model_specs:
        provider = get_provider(kind)
        out_path = out_dir / f"{spec.name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for task in tasks:
                output, latency = provider.complete(spec, task.prompt)
                s = score(task.scorer, output, task.reference)
                input_tokens = estimate_tokens(task.prompt)
                output_tokens = estimate_tokens(output)
                estimated_cost_usd = estimate_request_cost_usd(
                    input_tokens,
                    output_tokens,
                    spec.input_cost_per_million,
                    spec.output_cost_per_million,
                )
                record = {
                    "task_id": task.id,
                    "category": task.category,
                    "model": spec.name,
                    "output": output,
                    "score": s,
                    "latency_s": round(latency, 4),
                    "input_tokens_est": input_tokens,
                    "output_tokens_est": output_tokens,
                    "estimated_cost_usd": round(estimated_cost_usd, 8),
                    "input_cost_per_million": spec.input_cost_per_million,
                    "output_cost_per_million": spec.output_cost_per_million,
                }
                f.write(json.dumps(record) + "\n")
        print(f"[{spec.name}] wrote {len(tasks)} results -> {out_path}")
