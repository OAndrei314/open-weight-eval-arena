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


def test_task_from_dict_with_turns_is_multi_turn():
    task = Task.from_dict(
        {
            "id": "mt-x",
            "category": "multi_turn_consistency",
            "turns": ["turn one", "turn two"],
            "reference": "r",
            "scorer": "exact_match",
        }
    )
    assert task.is_multi_turn
    assert task.turns == ("turn one",)
    assert task.prompt == "turn two"
    assert task.all_turns == ("turn one", "turn two")


def test_task_from_dict_single_prompt_is_not_multi_turn():
    task = Task.from_dict(
        {"id": "s-x", "category": "c", "prompt": "p", "reference": "r", "scorer": "exact_match"}
    )
    assert not task.is_multi_turn
    assert task.turns == ()
    assert task.all_turns == ("p",)


def test_task_from_dict_rejects_both_prompt_and_turns():
    with pytest.raises(ValueError, match="either 'prompt' or 'turns'"):
        Task.from_dict(
            {
                "id": "x",
                "category": "c",
                "prompt": "p",
                "turns": ["a", "b"],
                "reference": "r",
                "scorer": "exact_match",
            }
        )


def test_task_from_dict_rejects_single_element_turns():
    with pytest.raises(ValueError, match="at least 2 messages"):
        Task.from_dict(
            {"id": "x", "category": "c", "turns": ["only one"], "reference": "r", "scorer": "exact_match"}
        )


def test_task_from_dict_rejects_missing_prompt_and_turns():
    with pytest.raises(ValueError, match="'prompt'.*or 'turns'"):
        Task.from_dict({"id": "x", "category": "c", "reference": "r", "scorer": "exact_match"})
