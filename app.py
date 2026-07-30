"""Streamlit dashboard.

    streamlit run app.py

Views: company view, cross-company view, semantic search, correlation
explorer. Reads from the SQLite DB + Chroma store populated by
`pipeline/runner.py` -- run that first if the tables are empty.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from store.db import init_db, get_session
from store.models import NlpResult, Transcript
from store.chroma_store import TranscriptStore
from analysis.correlations import run as run_correlations
from analysis.feature_builder import build_feature_table

st.set_page_config(page_title="earnings-lens", layout="wide")
init_db()


@st.cache_data(ttl=60)
def load_nlp_results() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.query(NlpResult).all()
        return pd.DataFrame([{c.name: getattr(r, c.name) for c in NlpResult.__table__.columns} for r in rows])
    finally:
        session.close()


@st.cache_data(ttl=60)
def load_transcripts_df() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.query(Transcript).all()
        return pd.DataFrame([{c.name: getattr(r, c.name) for c in Transcript.__table__.columns} for r in rows])
    finally:
        session.close()


st.title("earnings-lens")
st.caption(
    "Synthetic demo data -- see README limitations. NLP results were produced by fallback "
    "(lexicon/TF-IDF/spaCy-sm) models unless FinBERT/BERTopic were installed at pipeline run time."
)

df = load_nlp_results()
transcripts_df = load_transcripts_df()

if df.empty:
    st.warning("No results yet. Run `python pipeline/runner.py` first to populate the database.")
    st.stop()

df["period"] = df["quarter"] + " FY" + df["year"].astype(str)

tab_company, tab_cross, tab_search, tab_corr = st.tabs(
    ["Company view", "Cross-company view", "Semantic search", "Correlation explorer"]
)

with tab_company:
    tickers = sorted(df["ticker"].unique())
    ticker = st.selectbox("Company", tickers)
    cdf = df[df["ticker"] == ticker].sort_values(["year", "quarter"])

    st.subheader("Sentiment trend")
    fig = px.line(cdf, x="period", y="sentiment_weighted_score", markers=True, title="Weighted sentiment score (-1..1)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Topic evolution")
    topic_rows = []
    for _, row in cdf.iterrows():
        for t in row["top_topics"] or []:
            topic_rows.append({"period": row["period"], "topic": t["topic"], "weight": t["weight"]})
    if topic_rows:
        tdf = pd.DataFrame(topic_rows)
        fig2 = px.area(tdf, x="period", y="weight", color="topic", title="Topic weight over time")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tone radar")
    if not cdf.empty:
        latest = cdf.iloc[-1]
        radar_metrics = ["hedging_score", "certainty_score", "forward_looking_ratio", "avg_evasiveness"]
        values = [latest[m] or 0 for m in radar_metrics]
        fig3 = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=radar_metrics + [radar_metrics[0]], fill="toself"))
        fig3.update_layout(title=f"Tone profile - latest quarter ({latest['period']})")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Guidance tracker")
    guidance_rows = []
    for _, row in cdf.iterrows():
        for g in row["guidance_items"] or []:
            guidance_rows.append({"period": row["period"], **g})
    if guidance_rows:
        st.dataframe(pd.DataFrame(guidance_rows))
    st.line_chart(cdf.set_index("period")["guidance_accuracy"])

    st.subheader("Key quotes")
    trec = transcripts_df[transcripts_df["ticker"] == ticker].sort_values(["year", "quarter"])
    if not trec.empty:
        latest_t = trec.iloc[-1]
        st.write(latest_t["management_text"][:800] + ("..." if len(latest_t["management_text"]) > 800 else ""))

with tab_cross:
    st.subheader("Sector sentiment comparison")
    merged = df.merge(transcripts_df[["ticker", "quarter", "year", "sector"]], on=["ticker", "quarter", "year"], how="left")
    sector_sent = merged.groupby("sector", dropna=False)["sentiment_weighted_score"].mean().reset_index()
    st.plotly_chart(px.bar(sector_sent, x="sector", y="sentiment_weighted_score"), use_container_width=True)

    st.subheader("Hedging leaderboard")
    hedge_lb = df.groupby("ticker")["hedging_score"].mean().sort_values(ascending=False).reset_index()
    st.dataframe(hedge_lb)

    st.subheader("Guidance accuracy leaderboard")
    acc_lb = df.groupby("ticker")["guidance_accuracy"].mean().sort_values(ascending=False).reset_index()
    st.dataframe(acc_lb)

    st.subheader("Topic heatmap")
    topic_rows_all = []
    for _, row in df.iterrows():
        for t in row["top_topics"] or []:
            topic_rows_all.append({"ticker": row["ticker"], "topic": t["topic"], "weight": t["weight"]})
    if topic_rows_all:
        heat_df = pd.DataFrame(topic_rows_all).groupby(["ticker", "topic"])["weight"].mean().reset_index()
        pivot = heat_df.pivot(index="ticker", columns="topic", values="weight").fillna(0)
        st.plotly_chart(px.imshow(pivot, aspect="auto"), use_container_width=True)

with tab_search:
    st.subheader("Semantic search (Chroma)")
    query = st.text_input("Search transcripts (e.g. 'margin pressure and attrition')")
    if query:
        store = TranscriptStore()
        hits = store.search(query, n_results=5)
        for h in hits:
            st.markdown(f"**{h['metadata']['ticker']} {h['metadata']['quarter']} FY{h['metadata']['year']}** (distance={h['distance']:.4f})")
            st.write(h["text"][:400] + "...")
            st.divider()

with tab_corr:
    st.subheader("Correlation explorer")
    corr_df = run_correlations()
    st.dataframe(corr_df)

    feat_df = build_feature_table()
    if not feat_df.empty:
        feature = st.selectbox("Feature", [c for c in feat_df.columns if c not in ("ticker", "quarter", "year")])
        outcome = st.selectbox("Outcome", ["revenue_surprise_pct", "return_30d_pct", "revenue_growth_actual_pct"])
        fig4 = px.scatter(feat_df, x=feature, y=outcome, hover_data=["ticker", "quarter", "year"], trendline="ols")
        st.plotly_chart(fig4, use_container_width=True)
