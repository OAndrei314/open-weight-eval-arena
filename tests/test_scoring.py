from arena.scoring import exact_match, keyword_coverage, regex_rubric, score


def test_exact_match_case_insensitive():
    assert exact_match(" Paris ", "paris") == 1.0
    assert exact_match("London", "paris") == 0.0


def test_keyword_coverage_partial():
    assert keyword_coverage("I like power and land", "power, land, chips") == 2 / 3
    assert keyword_coverage("", "power, land, chips") == 0.0
    assert keyword_coverage("anything", "") == 0.0


def test_regex_rubric_matches():
    assert regex_rubric("retry", "^(retry|abort|escalate)$") == 1.0
    assert regex_rubric("maybe retry?", "^(retry|abort|escalate)$") == 0.0


def test_score_dispatch_unknown_scorer_raises():
    try:
        score("not_a_real_scorer", "x", "y")
        assert False, "expected ValueError"
    except ValueError:
        pass
