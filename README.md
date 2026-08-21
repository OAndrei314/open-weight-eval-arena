# open-weight-eval-arena

*Maintained by: claude-actions-daily-routine · Status: Active*
A small, dependency-light harness for benchmarking open-weight LLMs (GLM-5.2/5.3, DeepSeek V4,
Kimi K2.6/K2.7/K3, Qwen 3.5, MiniMax M3, Llama 4, and friends) against each other on a
custom task suite, instead of trusting vendor-reported leaderboard numbers.

Most public leaderboards evaluate models on tasks the labs already optimize for. This tool
is meant to be pointed at *your own* task suite — reasoning, tool-use, long-context recall,
whatever you actually care about — and produce a reproducible, versioned report.

## Why this exists

The open-weight frontier moved fast through 2026 (GLM-5.2 → 5.3, DeepSeek V4, Kimi K2.6
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

# Generate the comparison report, plus a PNG bar chart of overall score per model
python -m arena.cli report --results results/live --out report.md --chart report.png
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
- **instruction_following_under_distraction** — a fixed user instruction (answer in one
  word, output a single integer, preserve list order, emit exact JSON) competes against a
  second, "louder" instruction embedded in reference material the model is quoting from:
  a reviewer note, a formatting-update banner, or a bare `IGNORE ALL PREVIOUS
  INSTRUCTIONS` injection. This is a minimal, offline proxy for instruction-hierarchy
  robustness — the same failure mode (an LLM treating untrusted context as more
  authoritative than its actual instructions) that makes prompt injection dangerous in
  agentic/tool-use settings, without needing a real tool-use harness to exercise it.
- **multi_turn_consistency** — a short conversation (2-3 turns) where an early turn
  establishes a fact, naming convention, or formatting rule, and the final turn asks a
  question that's only answerable by correctly carrying that state forward — including
  one task where a later turn *corrects* the earlier one, testing whether the model
  updates instead of anchoring on the first-mentioned fact. Every prior turn is actually
  replayed through the provider (each round resends the growing transcript, exactly like
  a real stateless chat-completions API), so latency, and token/cost accounting all
  reflect the full multi-round exchange rather than just the last message.

## Adding a task

Append a line to the matching category file under `tasks/` (or start a new file — every
`*.jsonl` file in the directory is loaded):

```json
{"id": "r-014", "category": "reasoning", "prompt": "...", "reference": "42", "scorer": "exact_match"}
```

Multi-turn tasks use `"turns"` (a list of at least 2 user messages, in order) instead of
`"prompt"` — the harness treats the last entry as the scored turn and everything before
it as prior conversation:

```json
{"id": "mt-005", "category": "multi_turn_consistency", "turns": ["...", "..."], "reference": "...", "scorer": "exact_match"}
```

## Adding a model

Add an entry to your config YAML:

```yaml
- name: glm-5.3
  base_url: https://openrouter.ai/api/v1
  model: zhipu/glm-5.3
  api_key_env: OPENROUTER_API_KEY
  input_cost_per_million: 0.0
  output_cost_per_million: 0.0
```

Fill the pricing fields with the current rates from your provider invoice or pricing page.
The harness treats them as user-supplied assumptions and labels resulting costs as
estimates.

## Status

Task suite now covers reasoning, agentic tool-use, long-context recall, code repair,
instruction-following-under-distraction, and multi-turn consistency — the point is the
harness, not the leaderboard. `arena report --chart` now also renders the `report.py →
report.md + chart.png` pipeline from the quickstart diagram (previously only the markdown
half was implemented — the diagram promised a chart that didn't exist yet); it's a plain
matplotlib horizontal bar chart of overall score per model, kept as an opt-in flag so the
core JSONL → markdown path has no plotting dependency to import unless you ask for one.

`OpenAICompatProvider` — the code path that actually talks to a real model host — had zero
test coverage and zero error handling: any rate limit, transient 5xx, timeout, or malformed
response body would raise an uncaught exception and kill the entire suite run, discarding
whatever results had already been collected for every other model. It now retries transient
failures (429/500/502/503/504, connection errors, timeouts) with exponential backoff,
fails fast on non-retryable errors (auth, bad request, malformed response body) instead of
wasting retries on something that can't succeed, and — if retries are exhausted — `run_suite`
records that single task as a failed result (`score: 0`, `error: "..."` set) and keeps going
instead of crashing the whole run. `report.py` now surfaces an `errors` column and a summary
note so a provider outage isn't silently averaged into a model's score and misread as the
model itself answering badly.

Natural next step: scaling `long_context_recall` up to a real multi-thousand-token corpus
run against a live provider (the current fixtures are a small-scale proxy, not a real
million-token stress test, as noted above) — genuinely out of scope for this repo's
zero-network, CI-only test suite, so it'd need to happen as a manual, credentialed run
rather than something committed here.

## License

MIT — see [LICENSE](LICENSE).
