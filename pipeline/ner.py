"""Named-entity recognition + custom regex patterns for Indian regulators
and numeric claims.

Real path: spaCy `en_core_web_trf` if installed (transformer pipeline).
Fallback (and default in this repo, since the trf model is a multi-GB
download): spaCy `en_core_web_sm`. If spaCy itself isn't installed at all,
falls back further to the regex-only patterns below.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

REGULATOR_RE = re.compile(r"\b(SEBI|RBI|TRAI)\b")
NUMERIC_CLAIM_RE = re.compile(
    r"\b(?:Rs\.?\s?\d[\d,]*(?:\.\d+)?\s?(?:crore|lakh|cr)?|\d+(?:\.\d+)?\s?%)\b", re.IGNORECASE
)

_nlp = None
_spacy_load_failed = False
_MODEL_PREFERENCE = ("en_core_web_trf", "en_core_web_sm")


def _get_spacy_model():
    global _nlp, _spacy_load_failed
    if _nlp is not None or _spacy_load_failed:
        return _nlp
    try:
        import spacy

        for name in _MODEL_PREFERENCE:
            try:
                _nlp = spacy.load(name)
                return _nlp
            except OSError:
                continue
        _spacy_load_failed = True
    except Exception:
        _spacy_load_failed = True
    return _nlp


def extract_entities(text: str) -> list[dict]:
    """Return [{"text":..., "label":..., "start":..., "end":...}] using
    spaCy if available, else just the regulator/numeric regex hits.
    """
    entities = []
    nlp = _get_spacy_model()
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})

    for m in REGULATOR_RE.finditer(text):
        entities.append({"text": m.group(0), "label": "REGULATOR", "start": m.start(), "end": m.end()})
    for m in NUMERIC_CLAIM_RE.finditer(text):
        entities.append({"text": m.group(0), "label": "NUMERIC_CLAIM", "start": m.start(), "end": m.end()})

    return entities


def entity_frequency(entities: list[dict]) -> dict:
    counter = Counter((e["label"], e["text"]) for e in entities)
    freq = defaultdict(list)
    for (label, text), count in counter.items():
        freq[label].append({"text": text, "count": count})
    for label in freq:
        freq[label].sort(key=lambda d: -d["count"])
    return dict(freq)


def entity_sentiment_context(text: str, entities: list[dict], window: int = 60) -> list[dict]:
    """Sentiment-in-context per entity: score the local window of text
    around each entity mention using the sentiment fallback lexicon (cheap,
    always available, avoids re-triggering the heavy sentiment engine per
    entity mention).
    """
    from pipeline.sentiment import lexicon_sentence_score, _label_from_score

    results = []
    for e in entities:
        start = max(0, e["start"] - window)
        end = min(len(text), e["end"] + window)
        context = text[start:end]
        score = lexicon_sentence_score(context)
        results.append({**e, "context": context, "sentiment_score": round(score, 4), "label_sentiment": _label_from_score(score)})
    return results
