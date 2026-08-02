from pipeline.tone import (
    forward_looking_ratio,
    hedging_score,
    is_evasive,
    question_answer_similarity,
)


def test_on_topic_answer_has_higher_similarity_than_evasive_answer():
    question = "What is driving your margin improvement this quarter?"
    on_topic = "Margin improvement this quarter came from operating leverage and lower attrition costs."
    evasive = "We remain focused on our long-term vision and holistic strategic priorities."

    sim_on_topic = question_answer_similarity(question, on_topic)
    sim_evasive = question_answer_similarity(question, evasive)

    assert sim_on_topic > sim_evasive


def test_is_evasive_flags_low_similarity_answer():
    question = "What is your capex plan for next year?"
    evasive_answer = "There are a lot of moving parts and we'd rather not get into specifics."
    assert is_evasive(question, evasive_answer) is True


def test_is_evasive_false_for_on_topic_answer():
    question = "What is your capex plan for next year?"
    answer = "Our capex plan for next year focuses on capacity expansion in core segments."
    assert is_evasive(question, answer) is False


def test_hedging_score_counts_hedge_words():
    text = "It might possibly improve, though it could perhaps depend on conditions."
    assert hedging_score(text) > 0


def test_forward_looking_ratio_detects_future_modals():
    text = "We will expand capacity. We plan to hire more engineers. The weather was nice."
    ratio = forward_looking_ratio(text)
    assert 0 < ratio <= 1
