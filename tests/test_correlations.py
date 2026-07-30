import pandas as pd

from analysis.correlations import compute_correlations
from analysis.feature_builder import FEATURE_COLUMNS, OUTCOME_COLUMNS


def test_compute_correlations_returns_row_per_feature_outcome_pair():
    n = 10
    df = pd.DataFrame(
        {
            **{f: list(range(n)) for f in FEATURE_COLUMNS},
            **{o: list(range(n)) for o in OUTCOME_COLUMNS},
        }
    )
    df["guidance_met"] = [bool(i % 2) for i in range(n)]
    result = compute_correlations(df)
    assert len(result) == len(FEATURE_COLUMNS) * len(OUTCOME_COLUMNS)
    assert set(result.columns) >= {"feature", "outcome", "n", "rho", "p_value", "significant"}


def test_perfectly_correlated_columns_are_significant():
    n = 20
    df = pd.DataFrame({"sentiment_weighted_score": list(range(n)), "return_30d_pct": list(range(n))})
    for f in FEATURE_COLUMNS:
        if f != "sentiment_weighted_score":
            df[f] = 0
    for o in OUTCOME_COLUMNS:
        if o != "return_30d_pct":
            df[o] = 0
    df["guidance_met"] = True
    result = compute_correlations(df)
    row = result[(result["feature"] == "sentiment_weighted_score") & (result["outcome"] == "return_30d_pct")].iloc[0]
    assert row["rho"] == 1.0
    assert bool(row["significant"]) is True


def test_insufficient_data_reports_none_not_error():
    df = pd.DataFrame({"sentiment_weighted_score": [0.1], "return_30d_pct": [1.0]})
    for f in FEATURE_COLUMNS:
        if f not in df.columns:
            df[f] = 0.0
    for o in OUTCOME_COLUMNS:
        if o not in df.columns:
            df[o] = 0.0
    result = compute_correlations(df)
    assert result["rho"].isna().all() or (result["rho"] == None).all() or result["significant"].eq(False).all()
