from arena.providers import ModelSpec, MockProvider
from arena.scoring import score
from arena.tasks import load_tasks

MOCK_A = ModelSpec(name="mock-a", base_url="", model="mock-a")
MOCK_B = ModelSpec(name="mock-b", base_url="", model="mock-b")


def test_mock_provider_simulates_stronger_and_weaker_models():
    provider = MockProvider()
    prompt = (
        "A datacenter campus has 4 buildings, each drawing 60 MW at full load. "
        "If the site's grid interconnect is capped at 200 MW, what is the maximum "
        "number of buildings that can run at full load simultaneously?"
    )

    strong, _ = provider.complete(MOCK_A, prompt)
    weak, _ = provider.complete(MOCK_B, prompt)

    assert strong == "3"
    assert weak == "4"


def test_mock_provider_long_context_recall_strong_model_always_finds_needle():
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "long_context_recall"]
    assert len(tasks) >= 5

    for task in tasks:
        output, _ = provider.complete(MOCK_A, task.prompt)
        assert output == task.reference, f"{task.id}: strong model should always recall the needle"


def test_mock_provider_long_context_recall_weak_model_loses_middle_needles():
    """Reproduces the well-documented "lost in the middle" effect: recall near the
    edges of the context is reliable, recall from the middle third degrades — this
    is what makes the category informative rather than trivially all-pass/all-fail.
    """
    provider = MockProvider()
    tasks = {t.id: t for t in load_tasks("tasks") if t.category == "long_context_recall"}

    edge_output, _ = provider.complete(MOCK_B, tasks["lc-001"].prompt)  # needle near start
    assert edge_output == tasks["lc-001"].reference

    middle_output, _ = provider.complete(MOCK_B, tasks["lc-003"].prompt)  # needle near middle
    assert middle_output != tasks["lc-003"].reference

    late_edge_output, _ = provider.complete(MOCK_B, tasks["lc-005"].prompt)  # needle near end
    assert late_edge_output == tasks["lc-005"].reference


def test_mock_provider_code_repair_strong_model_fixes_every_bug():
    """The strong mock model should produce a fix that actually satisfies each task's
    own scorer — not just a hardcoded string that happens to look right. This would
    catch a typo'd reference regex or a mock answer that drifted out of sync with it.
    """
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "code_repair"]
    assert len(tasks) == 5

    for task in tasks:
        output, _ = provider.complete(MOCK_A, task.prompt)
        assert score(task.scorer, output, task.reference) == 1.0, (
            f"{task.id}: strong model's fix {output!r} should satisfy the scorer"
        )


def test_mock_provider_code_repair_weak_model_leaves_bugs_unfixed():
    """The weak mock model should reproduce a *plausible* failure mode (parroting the
    original buggy line) rather than trivially always failing, so the category actually
    discriminates between models instead of being all-pass/all-fail by construction.
    """
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "code_repair"]

    for task in tasks:
        output, _ = provider.complete(MOCK_B, task.prompt)
        assert score(task.scorer, output, task.reference) == 0.0, (
            f"{task.id}: weak model's output {output!r} should NOT satisfy the scorer"
        )


def test_mock_provider_ifd_strong_model_resists_distraction():
    """The strong mock model should follow the original instruction and ignore the
    embedded distractor/injection text on every task in the category, not just some.
    """
    provider = MockProvider()
    tasks = [
        t for t in load_tasks("tasks") if t.category == "instruction_following_under_distraction"
    ]
    assert len(tasks) == 5

    for task in tasks:
        output, _ = provider.complete(MOCK_A, task.prompt)
        assert score(task.scorer, output, task.reference) == 1.0, (
            f"{task.id}: strong model's output {output!r} should satisfy the scorer"
        )


def test_mock_provider_ifd_weak_model_gets_distracted():
    """The weak mock model should reproduce a *plausible* distraction failure (following
    the embedded note/injection instead of the original instruction) rather than just
    returning garbage, so the category discriminates instruction-hierarchy robustness
    specifically rather than general incompetence.
    """
    provider = MockProvider()
    tasks = [
        t for t in load_tasks("tasks") if t.category == "instruction_following_under_distraction"
    ]

    for task in tasks:
        output, _ = provider.complete(MOCK_B, task.prompt)
        assert score(task.scorer, output, task.reference) == 0.0, (
            f"{task.id}: weak model's output {output!r} should NOT satisfy the scorer"
        )


def test_mock_provider_ifd_injection_task_weak_model_outputs_injected_word():
    """Specifically check the classic "ignore all previous instructions" injection task
    (ifd-004): the weak model should comply with the injected instruction and output
    the word it demanded, not just fail the scorer for some unrelated reason.
    """
    provider = MockProvider()
    tasks = {
        t.id: t
        for t in load_tasks("tasks")
        if t.category == "instruction_following_under_distraction"
    }
    output, _ = provider.complete(MOCK_B, tasks["ifd-004"].prompt)
    assert output == "override"
