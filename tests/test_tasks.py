import pytest

from arena.tasks import Task, load_tasks


def test_load_tasks_from_repo_task_dir():
    tasks = load_tasks("tasks")
    assert len(tasks) >= 5
    ids = [t.id for t in tasks]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_task_from_dict_missing_field_raises():
    with pytest.raises(ValueError):
        Task.from_dict({"id": "x", "category": "c", "prompt": "p"})


def test_load_tasks_rejects_duplicate_ids(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        '{"id": "dup", "category": "c", "prompt": "p1", "reference": "r", "scorer": "exact_match"}\n'
        '{"id": "dup", "category": "c", "prompt": "p2", "reference": "r", "scorer": "exact_match"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate task ids"):
        load_tasks(tmp_path)
