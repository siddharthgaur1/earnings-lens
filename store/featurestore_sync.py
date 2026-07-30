"""Exports the NLP-derived feature table (analysis.feature_builder) into the
sibling "lite-featurestore" project as a registered FeatureGroup, entity_id
= ticker, event_timestamp = quarter-end date.

This turns per-transcript NLP signals (sentiment, hedging, evasiveness,
guidance accuracy, ...) into something a downstream model (e.g.
nifty-forecaster) could pull via featurestore's point-in-time-correct
`get_historical_features`, instead of re-running this whole NLP pipeline
itself.

featurestore is an optional dependency -- earnings-lens has no hard import
of it anywhere else; only this module (and its caller) needs it installed
(pip install -e ../featurestore, not published to PyPI).
"""
from __future__ import annotations

import pandas as pd

from analysis.feature_builder import FEATURE_COLUMNS, build_feature_table

FEATURE_GROUP_NAME = "earnings_call_nlp_features"

_QUARTER_END_MONTH_DAY = {"Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31)}


def quarter_end_date(quarter: str, year: int) -> pd.Timestamp:
    """Approximate the transcript's event date as the calendar quarter-end.
    Good enough for a demo entity_id/event_timestamp join key -- earnings
    calls actually happen a few weeks after quarter-end, but there's no real
    call date in the synthetic dataset to use instead."""
    month, day = _QUARTER_END_MONTH_DAY[quarter]
    return pd.Timestamp(year=year, month=month, day=day, tz="UTC")


def build_featurestore_frame(database_url: str | None = None) -> pd.DataFrame:
    """FEATURE_COLUMNS keyed by (entity_id=ticker, event_timestamp=quarter-end).
    Financial outcomes are deliberately excluded -- they're labels/outcomes
    for the correlation analysis, not predictive features to serve."""
    table = build_feature_table(database_url)
    if table.empty:
        return pd.DataFrame(columns=["entity_id", "event_timestamp"] + FEATURE_COLUMNS)

    out = table[["ticker", "quarter", "year"] + FEATURE_COLUMNS].copy()
    out["entity_id"] = out["ticker"]
    out["event_timestamp"] = [
        quarter_end_date(q, y) for q, y in zip(out["quarter"], out["year"])
    ]
    return out[["entity_id", "event_timestamp"] + FEATURE_COLUMNS]


def sync_to_feature_store(feature_store, database_url: str | None = None) -> dict:
    """Register `earnings_call_nlp_features` (idempotent -- safe to call on
    every pipeline run) and ingest the current feature table into it."""
    from featurestore import Entity, Feature, FeatureGroup

    frame = build_featurestore_frame(database_url)

    ticker_entity = Entity(name="ticker", dtype="string", description="NSE ticker")
    group = FeatureGroup(
        name=FEATURE_GROUP_NAME,
        entity=ticker_entity,
        features=[Feature(name=col, dtype="float32") for col in FEATURE_COLUMNS],
        ttl_hours=None,  # quarterly data, never expire in the online store
        online=True,
        offline=True,
        tags=["earnings-lens", "nlp", "quarterly"],
    )
    feature_store.register(group)

    if frame.empty:
        return {"rows_written": 0, "stats": {}}
    return feature_store.ingest(feature_group=FEATURE_GROUP_NAME, data=frame)
