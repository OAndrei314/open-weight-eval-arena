from arena.cost import estimate_request_cost_usd, estimate_tokens


def test_estimate_tokens_is_deterministic_and_nonzero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("alpha beta gamma") == 4


def test_estimate_request_cost_uses_input_and_output_rates():
    cost = estimate_request_cost_usd(
        input_tokens=1_000,
        output_tokens=2_000,
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
    )

    assert abs(cost - 0.0035) < 1e-12
