"""Joins nlp_results with financial_outcomes (both keyed on
ticker/quarter/year) into a single flat feature table for correlation
analysis and the dashboard.
"""
from __future__ import annotations

import pandas as pd

from store.db import get_session
from store.models import NlpResult, FinancialOutcome


FEATURE_COLUMNS = [
    "sentiment_weighted_score",
    "sentiment_pos_ratio",
    "sentiment_neg_ratio",
    "mgmt_vs_analyst_gap",
    "hedging_score",
    "certainty_score",
    "forward_looking_ratio",
    "flesch_kincaid_grade",
    "avg_evasiveness",
    "guidance_accuracy",
]

OUTCOME_COLUMNS = [
    "revenue_growth_actual_pct",
    "revenue_surprise_pct",
    "return_30d_pct",
    "guidance_met",
]


def build_feature_table(database_url: str | None = None) -> pd.DataFrame:
    session = get_session(database_url) if database_url else get_session()
    try:
        nlp_rows = session.query(NlpResult).all()
        outcome_rows = session.query(FinancialOutcome).all()
    finally:
        session.close()

    nlp_df = pd.DataFrame(
        [
            {c.name: getattr(row, c.name) for c in NlpResult.__table__.columns}
            for row in nlp_rows
        ]
    )
    outcome_df = pd.DataFrame(
        [
            {c.name: getattr(row, c.name) for c in FinancialOutcome.__table__.columns}
            for row in outcome_rows
        ]
    )
    if nlp_df.empty or outcome_df.empty:
        return pd.DataFrame(columns=["ticker", "quarter", "year"] + FEATURE_COLUMNS + OUTCOME_COLUMNS)

    merged = nlp_df.merge(outcome_df, on=["ticker", "quarter", "year"], suffixes=("", "_outcome"))
    keep = ["ticker", "quarter", "year"] + FEATURE_COLUMNS + OUTCOME_COLUMNS
    return merged[[c for c in keep if c in merged.columns]]
