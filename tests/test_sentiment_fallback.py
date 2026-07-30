from pipeline.sentiment import lexicon_sentence_score, score_sentences, management_vs_analyst_gap


def test_lexicon_positive_sentence():
    score = lexicon_sentence_score("We delivered strong growth and robust margins this quarter.")
    assert score > 0


def test_lexicon_negative_sentence():
    score = lexicon_sentence_score("Revenue declined amid weak demand and margin pressure.")
    assert score < 0


def test_lexicon_negation_flips_polarity():
    positive = lexicon_sentence_score("Growth was strong this quarter.")
    negated = lexicon_sentence_score("Growth was not strong this quarter.")
    assert negated < positive


def test_score_sentences_uses_lexicon_fallback_when_finbert_disabled():
    result = score_sentences("We had strong growth. Margins were weak due to cost pressure.", use_finbert=False)
    assert result.engine == "lexicon_fallback"
    assert len(result.sentence_scores) == 2
    assert -1.0 <= result.weighted_score <= 1.0


def test_management_vs_analyst_gap_flags_large_divergence():
    mgmt = "We delivered strong, robust, confident growth with record profit and healthy margins."
    qa = [{"question": "How is demand?", "answer": "Demand is weak, declining, with challenging headwinds and risk."}]
    result = management_vs_analyst_gap(mgmt, qa)
    assert result["flag"] is True
    assert result["gap"] > 0
