"""Forward-guidance extraction: rule-based (regex/spaCy) by default.

Spec's "Step 2" (Claude API structuring of ambiguous guidance sentences) is
implemented as a stub function that is NOT called anywhere in this repo's
pipeline, tests, or demos -- wiring it up requires an Anthropic API key and
would make live, paid API calls, which this project deliberately avoids.
"""
from __future__ import annotations

import re

METRIC_PATTERNS = {
    "revenue": re.compile(r"\brevenue\b", re.IGNORECASE),
    "margin": re.compile(r"\bmargin[s]?\b", re.IGNORECASE),
    "ebitda": re.compile(r"\bebitda\b", re.IGNORECASE),
    "growth": re.compile(r"\bgrowth\b", re.IGNORECASE),
}
DIRECTION_UP = re.compile(r"\b(rais(?:e|ing)|increase[d]?|higher|improv\w*|up|expand\w*)\b", re.IGNORECASE)
DIRECTION_DOWN = re.compile(r"\b(lower\w*|declin\w*|reduc\w*|down|revis(?:e|ing) .* down|cut)\b", re.IGNORECASE)
MAGNITUDE_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%")
MAGNITUDE_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
TIMEFRAME_RE = re.compile(
    r"\b(next year|full year|full-year|coming year|next quarter|next two quarters|"
    r"next fiscal year|this year|near term|near-term)\b",
    re.IGNORECASE,
)


def extract_guidance_structured(sentence: str) -> dict:
    """Rule-based structuring of a single guidance-flavoured sentence into
    {metric, direction, magnitude, timeframe, confidence}.

    `confidence` is a simple heuristic in [0, 1]: how many of the four
    fields we managed to fill in confidently via regex, since this is a
    rule-based extractor (not a model with real calibrated confidence).
    """
    metric = None
    for name, pattern in METRIC_PATTERNS.items():
        if pattern.search(sentence):
            metric = name
            break

    direction = None
    if DIRECTION_UP.search(sentence):
        direction = "up"
    elif DIRECTION_DOWN.search(sentence):
        direction = "down"

    magnitude = None
    range_match = MAGNITUDE_RANGE_RE.search(sentence)
    if range_match:
        magnitude = {"low": float(range_match.group(1)), "high": float(range_match.group(2))}
    else:
        single_match = MAGNITUDE_SINGLE_RE.search(sentence)
        if single_match:
            magnitude = {"low": float(single_match.group(1)), "high": float(single_match.group(1))}

    timeframe_match = TIMEFRAME_RE.search(sentence)
    timeframe = timeframe_match.group(0).lower() if timeframe_match else None

    fields_filled = sum(x is not None for x in (metric, direction, magnitude, timeframe))
    confidence = round(fields_filled / 4, 2)

    return {
        "sentence": sentence,
        "metric": metric,
        "direction": direction,
        "magnitude": magnitude,
        "timeframe": timeframe,
        "confidence": confidence,
    }


def extract_all_guidance(guidance_statements: list[str]) -> list[dict]:
    return [extract_guidance_structured(s) for s in guidance_statements]


def structure_guidance_with_claude(sentence: str) -> dict:  # pragma: no cover - deliberately unused stub
    """STUB. Not called anywhere in this repo. Would call the Claude API to
    structure ambiguous guidance sentences the regex extractor above can't
    confidently parse (confidence < some threshold). Left unimplemented on
    purpose: wiring this up costs the user money per call and requires an
    API key, which this project must not do without explicit go-ahead.
    """
    raise NotImplementedError(
        "structure_guidance_with_claude is a stub and intentionally not implemented. "
        "It would require a paid Anthropic API call, which is disabled in this project."
    )


def guidance_accuracy(extracted: list[dict], actual_outcome: dict) -> dict:
    """Compare extracted guidance direction/magnitude against a synthetic
    actual outcome record (from data/processed/financial_outcomes.csv) and
    produce an accuracy score.

    Accuracy heuristic: for each extracted guidance item with a known
    direction, check whether the sign of `revenue_growth_actual_pct`
    (relative to a flat 8% baseline) agrees with the guided direction.
    """
    if not extracted:
        return {"n_guidance_items": 0, "accuracy": None}

    baseline = 8.0
    actual_direction = "up" if actual_outcome.get("revenue_growth_actual_pct", baseline) > baseline else "down"

    directional_items = [g for g in extracted if g["direction"] in ("up", "down")]
    if not directional_items:
        return {"n_guidance_items": len(extracted), "accuracy": None}

    correct = sum(1 for g in directional_items if g["direction"] == actual_direction)
    accuracy = round(correct / len(directional_items), 4)

    return {
        "n_guidance_items": len(extracted),
        "n_directional_items": len(directional_items),
        "actual_direction": actual_direction,
        "accuracy": accuracy,
    }
