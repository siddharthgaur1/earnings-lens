"""Tone / communication-style analysis: hedging, certainty, evasiveness,
readability, verbosity, forward-looking tense.
"""
from __future__ import annotations

import re

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "somewhat", "potentially",
    "likely", "unlikely", "seem", "seems", "appear", "appears", "believe",
    "think", "suggest", "suggests", "roughly", "approximately", "around",
    "tend", "tends", "uncertain", "unclear",
}
CERTAINTY_WORDS = {
    "will", "definitely", "certainly", "clearly", "confident", "confirm",
    "confirmed", "guarantee", "guaranteed", "committed", "commit", "ensure",
    "always", "never", "must", "undoubtedly",
}
FUTURE_MODALS = {"will", "shall", "going to", "plan to", "expect to", "aim to", "intend to"}

_WORD_RE = re.compile(r"[a-zA-Z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def hedging_score(text: str) -> float:
    """Ratio of hedge words to total words."""
    tokens = _words(text)
    if not tokens:
        return 0.0
    hedges = sum(1 for t in tokens if t in HEDGE_WORDS)
    return round(hedges / len(tokens), 4)


def certainty_score(text: str) -> float:
    tokens = _words(text)
    if not tokens:
        return 0.0
    certain = sum(1 for t in tokens if t in CERTAINTY_WORDS)
    return round(certain / len(tokens), 4)


def forward_looking_ratio(text: str) -> float:
    """Share of sentences containing a future-tense modal/phrase."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return 0.0
    text_l_sentences = [s.lower() for s in sentences]
    hits = sum(1 for s in text_l_sentences if any(m in s for m in FUTURE_MODALS))
    return round(hits / len(sentences), 4)


def verbosity(answer: str) -> int:
    """Words per answer."""
    return len(_words(answer))


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level. Uses `textstat` if installed (more
    accurate syllable counting), else a direct implementation of the
    standard formula with a simple vowel-group syllable heuristic.
    """
    try:
        import textstat

        return round(textstat.flesch_kincaid_grade(text), 2)
    except Exception:
        return round(_fk_grade_manual(text), 2)


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _fk_grade_manual(text: str) -> float:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    words = _words(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


def question_answer_similarity(question: str, answer: str) -> float:
    """TF-IDF cosine similarity between a question and its answer.

    This is the practical implementation of the evasiveness signal: a
    genuinely on-topic answer shares vocabulary with the question; a vague,
    deflecting answer usually doesn't. Upgrade path: swap in transformer
    sentence embeddings (e.g. sentence-transformers) for semantic (not just
    lexical) similarity -- see store/chroma_store.py which already has that
    fallback pattern.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not question.strip() or not answer.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([question, answer])
    except ValueError:
        # empty vocabulary after stop-word removal (e.g. very short strings)
        return 0.0
    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(float(sim), 4)


def is_evasive(question: str, answer: str, threshold: float = 0.08) -> bool:
    """Flag low question-answer lexical overlap as (candidate) evasiveness."""
    return question_answer_similarity(question, answer) < threshold


def analyze_qa_tone(qa: list[dict]) -> list[dict]:
    """Per Q&A pair: similarity, evasive flag, hedging/certainty of the
    answer, verbosity, forward-looking ratio.
    """
    results = []
    for pair in qa:
        q, a = pair["question"], pair["answer"]
        sim = question_answer_similarity(q, a)
        results.append(
            {
                "question": q,
                "answer": a,
                "qa_similarity": sim,
                "evasive": sim < 0.08,
                "hedging_score": hedging_score(a),
                "certainty_score": certainty_score(a),
                "verbosity": verbosity(a),
                "forward_looking_ratio": forward_looking_ratio(a),
                "flesch_kincaid_grade": flesch_kincaid_grade(a),
            }
        )
    return results
