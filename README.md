# open-weight-eval-arena

*Maintained by: claude-actions-daily-routine · Status: Active*
A small, dependency-light harness for benchmarking open-weight LLMs (GLM-5.2, DeepSeek V4,
Kimi K2.6/K2.7/K3, Qwen 3.5, MiniMax M3, Llama 4, and friends) against each other on a
custom task suite, instead of trusting vendor-reported leaderboard numbers.

Most public leaderboards evaluate models on tasks the labs already optimize for. This tool
is meant to be pointed at *your own* task suite — reasoning, tool-use, long-context recall,
whatever you actually care about — and produce a reproducible, versioned report.

## Why this exists

The open-weight frontier moved fast in the first half of 2026 (GLM-5.2, DeepSeek V4, Kimi K2.6
→ K2.7 → K3, MiniMax M3, Qwen 3.5), and picking a model for a given workload increasingly means
running your own evals rather than reading a benchmark table. This is a minimal, auditable
version of that process: plain JSONL task files, pluggable scorers, pluggable model providers,
markdown + chart output, and estimated cost-normalized ranking.

## How it works

```
tasks/*.jsonl        →  runner.py  →  results/*.jsonl  →  report.py  →  report.md + chart.png
configs/models.yaml  ─┘
```

- **Tasks** are plain JSONL: `{"id", "category", "prompt", "reference", "scorer"}`.
- **Providers** are anything exposing an OpenAI-compatible `/v1/chat/completions` endpoint —
  which covers most open-weight model hosts (OpenRouter, Fireworks, Together, Groq, local
  vLLM / Ollama / llama.cpp servers). A `MockProvider` is included so the full pipeline runs
  and is tested with zero API keys and zero network access.
- **Scorers** are simple and inspectable on purpose: exact-match, keyword-coverage, and
  regex-rubric. No LLM-judge-grading-LLM circularity here.
- **Cost estimates** use configurable USD-per-million-token rates plus deterministic token
  estimates, so reports can compare score, latency, and score-per-dollar instead of raw
  score alone.

## Quickstart

```bash
pip install -r requirements.txt

# Dry run against the built-in mock provider (no API keys needed)
python -m arena.cli run --config configs/mock.yaml --tasks tasks/ --out results/mock

# Real run: fill in configs/models.example.yaml with base_url + model name per provider,
# export the relevant API key env vars, then:
python -m arena.cli run --config configs/models.example.yaml --tasks tasks/ --out results/live

# Generate the comparison report
python -m arena.cli report --results results/live --out report.md
```

## Task categories

- **reasoning** — arithmetic/logic word problems with a single verifiable answer.
- **agentic_tool_use** — tool-selection and tool-call-ordering questions against a small
  fixed toolset.
- **long_context_recall** — needle-in-haystack: a short fact is buried inside a longer
  synthetic operations-report prompt (~800 words / ~1k tokens) at a controlled position,
  and the model must recall it exactly. This is a small-scale proxy, not a real
  million-token stress test — genuinely exercising a 1M-token window (as advertised for
  e.g. Kimi K3 and GLM-5.2) needs a much larger corpus and real API calls, which is out of
  scope for a zero-network, CI-friendly harness. `MockProvider`'s weaker model reproduces
  the "lost in the middle" pattern reported in the long-context literature: reliable
  recall when the needle sits near the start or end of the prompt, degraded recall when
  it's buried in the middle third.
- **code_repair** — a short buggy function plus a description of the symptom; the model
  must output the single corrected line. Covers common real-world bug classes (off-by-one
  loop bound, boundary-condition comparison operator, missing zero-division guard, mutable
  default argument, wrong variable returned) rather than full-file patches, so it stays
  scorable with a plain regex instead of needing code execution.

## Adding a task

Append a line to the matching category file under `tasks/` (or start a new file — every
`*.jsonl` file in the directory is loaded):

```json
{"id": "r-014", "category": "reasoning", "prompt": "...", "reference": "42", "scorer": "exact_match"}
```

## Adding a model

Add an entry to your config YAML:

```yaml
- name: glm-5.2
  base_url: https://openrouter.ai/api/v1
  model: zhipu/glm-5.2
  api_key_env: OPENROUTER_API_KEY
  input_cost_per_million: 0.0
  output_cost_per_million: 0.0
```

Fill the pricing fields with the current rates from your provider invoice or pricing page.
The harness treats them as user-supplied assumptions and labels resulting costs as
estimates.

## Status

Task suite now covers reasoning, agentic tool-use, long-context recall, and code repair —
the point is the harness, not the leaderboard. PRs adding more categories
(instruction-following-under-distraction is the next natural one) or scaling
`long_context_recall` up to a real multi-thousand-token corpus run against a live provider
are the natural next steps.

## License

MIT — see [LICENSE](LICENSE).
