import json
import urllib.error

import pytest

from arena.providers import ModelSpec, MockProvider, OpenAICompatProvider, ProviderError
from arena.scoring import score
from arena.tasks import load_tasks

MOCK_A = ModelSpec(name="mock-a", base_url="", model="mock-a")
MOCK_B = ModelSpec(name="mock-b", base_url="", model="mock-b")
REAL_SPEC = ModelSpec(name="real-model", base_url="https://example.invalid/v1", model="real-model")


class _FakeResponse:
    """Stands in for the `http.client.HTTPResponse` context manager `urlopen` returns."""

    def __init__(self, body: dict):
        self._raw = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._raw


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid/v1/chat/completions", code, "err", {}, None)


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


def test_mock_provider_agentic_tool_use_strong_model_answers_correctly():
    """Every agentic_tool_use task should be solvable by the strong mock model.

    Regression test for a substring-matching bug: the generic "sparse
    autoencoders" branch (meant for the r-005 reasoning task) used to shadow
    the more specific "latest paper on sparse autoencoders" branch (meant for
    a-001), because a-001's prompt also contains "sparse autoencoders" as a
    substring and the generic check ran first. That made the strong mock model
    score 0.0 on a-001 even though it should trivially pass.
    """
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "agentic_tool_use"]
    assert len(tasks) == 4

    for task in tasks:
        output, _ = provider.complete(MOCK_A, task.prompt)
        assert score(task.scorer, output, task.reference) == 1.0, (
            f"{task.id}: strong model's output {output!r} should satisfy the scorer"
        )


def test_mock_provider_agentic_tool_use_weak_model_discriminates():
    """The weak mock model should fail every task where a wrong-but-plausible
    answer exists (a-001 tool ordering, a-003 safe read/write, a-004 timeout
    handling). a-002 has a single unambiguous correct answer (there's only one
    sensible JSON key to read) so both tiers legitimately agree there.
    """
    provider = MockProvider()
    tasks = {t.id: t for t in load_tasks("tasks") if t.category == "agentic_tool_use"}

    for tid in ("a-001", "a-003", "a-004"):
        output, _ = provider.complete(MOCK_B, tasks[tid].prompt)
        assert score(tasks[tid].scorer, output, tasks[tid].reference) == 0.0, (
            f"{tid}: weak model's output {output!r} should NOT satisfy the scorer"
        )

    output, _ = provider.complete(MOCK_B, tasks["a-002"].prompt)
    assert score(tasks["a-002"].scorer, output, tasks["a-002"].reference) == 1.0


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


def test_mock_provider_multi_turn_strong_model_tracks_full_history():
    """The strong mock model should correctly answer every multi-turn task by using the
    full transcript, not just the final turn."""
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "multi_turn_consistency"]
    assert len(tasks) == 4

    for task in tasks:
        output, _, transcript = provider.complete_conversation(MOCK_A, list(task.all_turns))
        assert output == task.reference, f"{task.id}: strong model should track full history"
        assert transcript[0] == {"role": "user", "content": task.all_turns[0]}
        assert transcript[-2]["role"] == "user"
        assert transcript[-1]["role"] == "assistant"


def test_mock_provider_multi_turn_weak_model_loses_earlier_context():
    """The weak mock model should reproduce plausible long-conversation failure modes
    (forgetting an earlier instruction, anchoring on a stale fact) rather than trivially
    always failing, so the category actually discriminates instead of being all-or-nothing."""
    provider = MockProvider()
    tasks = [t for t in load_tasks("tasks") if t.category == "multi_turn_consistency"]

    for task in tasks:
        output, _, _ = provider.complete_conversation(MOCK_B, list(task.all_turns))
        assert score(task.scorer, output, task.reference) == 0.0, (
            f"{task.id}: weak model's output {output!r} should NOT satisfy the scorer"
        )


def test_mock_provider_multi_turn_correction_task_weak_model_anchors_on_stale_fact():
    """Specifically check mt-003: the weak model should answer with the first-mentioned
    name (Priya) rather than the corrected one (Marcus), showing primacy-anchoring
    rather than an unrelated failure."""
    provider = MockProvider()
    tasks = {t.id: t for t in load_tasks("tasks") if t.category == "multi_turn_consistency"}
    output, _, _ = provider.complete_conversation(MOCK_B, list(tasks["mt-003"].all_turns))
    assert output == "priya"


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


def test_openai_compat_provider_parses_successful_response(monkeypatch):
    monkeypatch.setattr(
        "arena.providers.urllib.request.urlopen",
        lambda req, timeout: _FakeResponse({"choices": [{"message": {"content": "hello"}}]}),
    )
    provider = OpenAICompatProvider()
    text, latency = provider.complete(REAL_SPEC, "hi")
    assert text == "hello"
    assert latency >= 0.0


def test_openai_compat_provider_retries_transient_http_error_then_succeeds(monkeypatch):
    queue = [_http_error(503), _FakeResponse({"choices": [{"message": {"content": "ok"}}]})]

    def fake_urlopen(req, timeout):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("arena.providers.urllib.request.urlopen", fake_urlopen)
    sleeps = []
    provider = OpenAICompatProvider(max_retries=3, backoff_base_s=0.01, sleep_fn=sleeps.append)

    text, _ = provider.complete(REAL_SPEC, "hi")

    assert text == "ok"
    assert sleeps == [0.01]  # exactly one backoff, for the single retried attempt


def test_openai_compat_provider_retries_connection_error_then_succeeds(monkeypatch):
    queue = [urllib.error.URLError("connection refused"), _FakeResponse({"choices": [{"message": {"content": "ok"}}]})]

    def fake_urlopen(req, timeout):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("arena.providers.urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatProvider(backoff_base_s=0.0, sleep_fn=lambda s: None)

    text, _ = provider.complete(REAL_SPEC, "hi")

    assert text == "ok"


def test_openai_compat_provider_raises_provider_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(
        "arena.providers.urllib.request.urlopen", lambda req, timeout: (_ for _ in ()).throw(_http_error(500))
    )
    sleeps = []
    provider = OpenAICompatProvider(max_retries=2, backoff_base_s=0.0, sleep_fn=sleeps.append)

    with pytest.raises(ProviderError, match="after 3 attempts"):
        provider.complete(REAL_SPEC, "hi")

    assert sleeps == [0.0, 0.0]  # backoff attempted before each retry, not after the final failure


def test_openai_compat_provider_fails_fast_on_non_retryable_http_error(monkeypatch):
    monkeypatch.setattr(
        "arena.providers.urllib.request.urlopen", lambda req, timeout: (_ for _ in ()).throw(_http_error(401))
    )
    sleeps = []
    provider = OpenAICompatProvider(max_retries=3, sleep_fn=sleeps.append)

    with pytest.raises(ProviderError, match="401"):
        provider.complete(REAL_SPEC, "hi")

    assert sleeps == []  # no retries wasted on a request that will never succeed


def test_openai_compat_provider_raises_provider_error_on_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "arena.providers.urllib.request.urlopen",
        lambda req, timeout: _FakeResponse({"unexpected": "shape"}),
    )
    sleeps = []
    provider = OpenAICompatProvider(sleep_fn=sleeps.append)

    with pytest.raises(ProviderError, match="malformed response"):
        provider.complete(REAL_SPEC, "hi")

    assert sleeps == []  # a malformed response body won't fix itself on retry


def test_openai_compat_provider_complete_conversation_propagates_provider_error(monkeypatch):
    monkeypatch.setattr(
        "arena.providers.urllib.request.urlopen", lambda req, timeout: (_ for _ in ()).throw(_http_error(401))
    )
    provider = OpenAICompatProvider()

    with pytest.raises(ProviderError):
        provider.complete_conversation(REAL_SPEC, ["turn one", "turn two"])
