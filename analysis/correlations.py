"""Spearman correlations between NLP features and financial outcomes.

Reports every feature x outcome pair, including non-significant ones --
this is meant to be run for real against the synthetic dataset and its
actual output quoted in the README, not cherry-picked.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from analysis.feature_builder import (
    FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    build_feature_table,
)


def compute_correlations(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue
        for outcome in OUTCOME_COLUMNS:
            if outcome not in df.columns:
                continue
            sub = df[[feature, outcome]].dropna()
            # guidance_met is boolean; Spearman handles it fine as 0/1
            sub_outcome = sub[outcome].astype(float) if sub[outcome].dtype == bool else sub[outcome]
            if len(sub) < 3 or sub[feature].nunique() < 2 or sub_outcome.nunique() < 2:
                rows.append(
                    {"feature": feature, "outcome": outcome, "n": len(sub), "rho": None, "p_value": None, "significant": False}
                )
                continue
            rho, p = spearmanr(sub[feature], sub_outcome)
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n": len(sub),
                    "rho": round(float(rho), 4),
                    "p_value": round(float(p), 4),
                    "significant": bool(p < alpha),
                }
            )
    return pd.DataFrame(rows)


def run(database_url: str | None = None) -> pd.DataFrame:
    df = build_feature_table(database_url)
    return compute_correlations(df)


if __name__ == "__main__":
    result = run()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    print(result.to_string(index=False))
    n_sig = int(result["significant"].sum())
    print(f"\n{n_sig} of {len(result)} feature-outcome pairs significant at alpha=0.05")
