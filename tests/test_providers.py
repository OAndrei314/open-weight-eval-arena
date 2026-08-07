from arena.providers import ModelSpec, MockProvider


def test_mock_provider_simulates_stronger_and_weaker_models():
    provider = MockProvider()
    prompt = (
        "A datacenter campus has 4 buildings, each drawing 60 MW at full load. "
        "If the site's grid interconnect is capped at 200 MW, what is the maximum "
        "number of buildings that can run at full load simultaneously?"
    )

    strong, _ = provider.complete(ModelSpec(name="mock-a", base_url="", model="mock-a"), prompt)
    weak, _ = provider.complete(ModelSpec(name="mock-b", base_url="", model="mock-b"), prompt)

    assert strong == "3"
    assert weak == "4"
