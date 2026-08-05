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

    @staticmethod
    def from_dict(d: dict) -> "Task":
        required = {"id", "category", "prompt", "reference", "scorer"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"task {d.get('id', '<unknown>')} missing fields: {missing}")
        return Task(
            id=d["id"],
            category=d["category"],
            prompt=d["prompt"],
            reference=d["reference"],
            scorer=d["scorer"],
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
