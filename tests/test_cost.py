from arena.cost import estimate_conversation_tokens, estimate_request_cost_usd, estimate_tokens


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


def test_estimate_conversation_tokens_resends_full_history_each_round():
    # Turn 1: user "a b" (2 tok) -> assistant "c" (1 tok)
    # Turn 2: user resends "a b" + "c" + new "d e f" (5 tok) -> assistant "g" (1 tok)
    transcript = [
        {"role": "user", "content": "a b"},
        {"role": "assistant", "content": "c"},
        {"role": "user", "content": "d e f"},
        {"role": "assistant", "content": "g"},
    ]

    input_tokens, output_tokens = estimate_conversation_tokens(transcript)

    turn1_input = estimate_tokens("a b")
    turn2_input = estimate_tokens("a b") + estimate_tokens("c") + estimate_tokens("d e f")
    assert input_tokens == turn1_input + turn2_input
    assert output_tokens == estimate_tokens("c") + estimate_tokens("g")


def test_estimate_conversation_tokens_exceeds_naive_per_message_sum():
    """Cumulative resend accounting must charge more than summing each message once,
    since earlier turns get billed again on every later round."""
    transcript = [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "and now a follow up question"},
        {"role": "assistant", "content": "an answer"},
    ]
    naive_sum = sum(estimate_tokens(m["content"]) for m in transcript)

    input_tokens, output_tokens = estimate_conversation_tokens(transcript)

    assert input_tokens + output_tokens > naive_sum
