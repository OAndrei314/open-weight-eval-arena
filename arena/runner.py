"""Runs a task suite against a list of model configs and writes JSONL results."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .cost import estimate_conversation_tokens, estimate_request_cost_usd, estimate_tokens
from .providers import ModelSpec, ProviderError, get_provider
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

    names = [spec.name for _, spec in specs]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        # Two models sharing a display name would silently overwrite each other's
        # `{name}.jsonl` result file in run_suite -- one model's entire evaluation
        # would vanish from the report with no error, exactly the failure mode
        # `arena.tasks.load_tasks` already guards against for duplicate task ids.
        raise ValueError(f"duplicate model names in {config_path}: {sorted(dupes)}")
    return specs


def run_suite(
    tasks: list[Task], model_specs: list[tuple[str, ModelSpec]], out_dir: str | Path
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, spec in model_specs:
        provider = get_provider(kind)
        out_path = out_dir / f"{spec.name}.jsonl"
        errors = 0
        with out_path.open("w", encoding="utf-8") as f:
            for task in tasks:
                error: str | None = None
                try:
                    if task.is_multi_turn:
                        output, latency, transcript = provider.complete_conversation(
                            spec, list(task.all_turns)
                        )
                        input_tokens, output_tokens = estimate_conversation_tokens(transcript)
                    else:
                        output, latency = provider.complete(spec, task.prompt)
                        input_tokens = estimate_tokens(task.prompt)
                        output_tokens = estimate_tokens(output)
                    s = score(task.scorer, output, task.reference)
                except ProviderError as e:
                    # A provider failure (rate limit exhausted, malformed response,
                    # auth error, ...) is an infrastructure failure, not the model
                    # answering wrong -- record it distinctly (score 0, error set)
                    # instead of letting it crash the whole suite run.
                    output, latency, input_tokens, output_tokens, s = "", 0.0, 0, 0, 0.0
                    error = str(e)
                    errors += 1
                    print(f"[{spec.name}] task {task.id} failed: {error}")
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
                    "error": error,
                }
                f.write(json.dumps(record) + "\n")
        suffix = f" ({errors} provider error(s))" if errors else ""
        print(f"[{spec.name}] wrote {len(tasks)} results -> {out_path}{suffix}")
