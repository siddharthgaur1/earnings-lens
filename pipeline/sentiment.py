"""Sentence-level + aggregate sentiment scoring.

Real path: FinBERT (ProsusAI/finbert) via `transformers`, used automatically
if the library AND model weights are available (weights are pulled from
the HF hub the first time -- if that download fails or is unavailable
offline, we fall back automatically and log why).

Fallback path: a small but real finance-domain word-polarity lexicon
(not a coin flip) with negation handling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- Fallback: finance-domain polarity lexicon -----------------------------
POSITIVE_WORDS = {
    "growth", "strong", "robust", "healthy", "improve", "improved", "improvement",
    "expansion", "expand", "beat", "outperform", "record", "confident", "confidence",
    "momentum", "profit", "profitable", "efficient", "efficiency", "leadership",
    "resilient", "upgrade", "upside", "stabilise", "stabilised", "stabilized",
    "win", "wins", "growing", "gain", "gains", "positive", "solid", "accelerate",
}
NEGATIVE_WORDS = {
    "decline", "weak", "weaker", "headwind", "headwinds", "pressure", "pressured",
    "challenge", "challenging", "miss", "underperform", "loss", "losses", "cautious",
    "uncertain", "uncertainty", "volatility", "volatile", "slowdown", "delay", "delayed",
    "attrition", "downgrade", "downside", "muted", "soft", "softer", "risk", "risks",
    "difficult", "concern", "concerns", "shortfall",
}
NEGATIONS = {"not", "no", "never", "without", "n't"}


def _tokenize(sentence: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", sentence.lower())


def lexicon_sentence_score(sentence: str) -> float:
    """Return a score in [-1, 1] for one sentence using the finance lexicon,
    with simple negation flipping for the token immediately preceding a
    polarity word.
    """
    tokens = _tokenize(sentence)
    if not tokens:
        return 0.0
    score = 0
    hits = 0
    for i, tok in enumerate(tokens):
        polarity = 0
        if tok in POSITIVE_WORDS:
            polarity = 1
        elif tok in NEGATIVE_WORDS:
            polarity = -1
        if polarity != 0:
            negated = i > 0 and tokens[i - 1] in NEGATIONS
            if negated:
                polarity *= -1
            score += polarity
            hits += 1
    if hits == 0:
        return 0.0
    return max(-1.0, min(1.0, score / max(hits, 1)))


def _label_from_score(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


# --- Real path: FinBERT -----------------------------------------------------
_finbert_pipeline = None
_finbert_load_failed = False


def _get_finbert():
    global _finbert_pipeline, _finbert_load_failed
    if _finbert_pipeline is not None or _finbert_load_failed:
        return _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline

        _finbert_pipeline = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        # covers: transformers not installed, model weights not
        # downloadable (offline / no network), or any runtime load error.
        _finbert_load_failed = True
        _finbert_pipeline = None
    return _finbert_pipeline


@dataclass
class SentimentResult:
    engine: str
    sentence_scores: list = field(default_factory=list)  # [{"sentence": str, "label": str, "score": float}]
    pos_ratio: float = 0.0
    neu_ratio: float = 0.0
    neg_ratio: float = 0.0
    weighted_score: float = 0.0  # -1..1


def score_sentences(text: str, use_finbert: bool = True) -> SentimentResult:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return SentimentResult(engine="none")

    finbert = _get_finbert() if use_finbert else None
    if finbert is not None:
        engine = "finbert"
        raw = finbert(sentences, truncation=True)
        results = []
        for sent, r in zip(sentences, raw):
            label = r["label"].lower()
            signed = {"positive": 1, "neutral": 0, "negative": -1}.get(label, 0) * r["score"]
            results.append({"sentence": sent, "label": label, "score": round(float(signed), 4)})
    else:
        engine = "lexicon_fallback"
        results = []
        for sent in sentences:
            score = lexicon_sentence_score(sent)
            results.append({"sentence": sent, "label": _label_from_score(score), "score": round(score, 4)})

    n = len(results)
    pos = sum(1 for r in results if r["label"] == "positive") / n
    neg = sum(1 for r in results if r["label"] == "negative") / n
    neu = 1 - pos - neg
    weighted = sum(r["score"] for r in results) / n

    return SentimentResult(
        engine=engine,
        sentence_scores=results,
        pos_ratio=round(pos, 4),
        neu_ratio=round(neu, 4),
        neg_ratio=round(neg, 4),
        weighted_score=round(weighted, 4),
    )


def management_vs_analyst_gap(management_text: str, qa: list[dict], use_finbert: bool = False) -> dict:
    """Compare management commentary sentiment vs the sentiment of
    management's Q&A answers; flag a meaningful gap (possible spin between
    prepared remarks and off-the-cuff answers).

    `use_finbert` defaults to False here (unlike `score_sentences`) because
    this is invoked twice per transcript from the pipeline runner across a
    thread pool -- defaulting to the heavy model would silently trigger a
    multi-hundred-MB download the first time this module is imported.
    """
    mgmt = score_sentences(management_text, use_finbert=use_finbert)
    answers_text = " ".join(pair["answer"] for pair in qa)
    qa_sent = score_sentences(answers_text, use_finbert=use_finbert) if answers_text.strip() else SentimentResult(engine=mgmt.engine)

    gap = round(mgmt.weighted_score - qa_sent.weighted_score, 4)
    return {
        "management_score": mgmt.weighted_score,
        "qa_score": qa_sent.weighted_score,
        "gap": gap,
        "flag": abs(gap) > 0.3,
        "engine": mgmt.engine,
    }
