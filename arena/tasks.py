"""Task loading. Tasks are plain JSONL so they're easy to diff and review in PRs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    reference: str
    scorer: str
    turns: tuple[str, ...] = ()  # prior conversation turns; empty for single-turn tasks

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 0

    @property
    def all_turns(self) -> tuple[str, ...]:
        """Full conversation as a flat tuple of user messages, `prompt` last."""
        return self.turns + (self.prompt,)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        required = {"id", "category", "reference", "scorer"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"task {d.get('id', '<unknown>')} missing fields: {missing}")
        task_id = d["id"]

        if "turns" in d and "prompt" in d:
            raise ValueError(f"task {task_id}: specify either 'prompt' or 'turns', not both")

        if "turns" in d:
            turns_field = d["turns"]
            if not isinstance(turns_field, list) or len(turns_field) < 2:
                raise ValueError(f"task {task_id}: 'turns' must be a list of at least 2 messages")
            if not all(isinstance(t, str) and t.strip() for t in turns_field):
                raise ValueError(f"task {task_id}: every entry in 'turns' must be a non-empty string")
            prior, prompt = tuple(turns_field[:-1]), turns_field[-1]
        elif "prompt" in d:
            prior, prompt = (), d["prompt"]
        else:
            raise ValueError(f"task {task_id}: must specify 'prompt' (single-turn) or 'turns' (multi-turn)")

        return Task(
            id=task_id,
            category=d["category"],
            prompt=prompt,
            reference=d["reference"],
            scorer=d["scorer"],
            turns=prior,
        )


def load_tasks(tasks_dir: str | Path) -> list[Task]:
    """Load every *.jsonl file in a directory into a flat list of Task, sorted by id."""
    tasks_dir = Path(tasks_dir)
    tasks: list[Task] = []
    for path in sorted(tasks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no} invalid JSON: {e}") from e
                tasks.append(Task.from_dict(d))
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate task ids across files: {dupes}")
    return sorted(tasks, key=lambda t: t.id)
