from arena.providers import ModelSpec, MockProvider
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
