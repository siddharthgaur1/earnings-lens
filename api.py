"""FastAPI backend.

    uvicorn api:app --reload

Endpoints:
    POST /analyze          {ticker, quarter, year}  -- run the pipeline for one transcript
    GET  /results/{ticker}/{quarter}                -- stored NLP results (quarter like "Q1-2024")
    GET  /compare                                   -- cross-company summary table
    POST /search           {query, n_results}       -- Chroma semantic search
    GET  /correlations                              -- Spearman feature/outcome correlations
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analysis.correlations import run as run_correlations
from pipeline.runner import load_outcomes, load_transcripts, process_one
from store.chroma_store import TranscriptStore
from store.db import get_session, init_db
from store.models import NlpResult

app = FastAPI(title="earnings-lens API")


class AnalyzeRequest(BaseModel):
    ticker: str
    quarter: str
    year: int


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


@app.on_event("startup")
def _startup():
    init_db()


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    records = load_transcripts(ticker=req.ticker)
    record = next((r for r in records if r["quarter"] == req.quarter and r["year"] == req.year), None)
    if record is None:
        raise HTTPException(404, f"No synthetic transcript found for {req.ticker} {req.quarter} {req.year}")
    outcomes = load_outcomes()
    result = process_one(record, outcomes.get((req.ticker, req.quarter, req.year)))
    return result["nlp_row"]


@app.get("/results/{ticker}/{quarter}")
def get_results(ticker: str, quarter: str):
    """`quarter` path segment format: Q1-2024"""
    try:
        q, year = quarter.split("-")
        year = int(year)
    except ValueError:
        raise HTTPException(400, "quarter must be formatted like Q1-2024")

    session = get_session()
    try:
        row = session.query(NlpResult).filter_by(ticker=ticker, quarter=q, year=year).first()
        if row is None:
            raise HTTPException(404, "No results found; run /analyze or the pipeline runner first")
        return {c.name: getattr(row, c.name) for c in NlpResult.__table__.columns}
    finally:
        session.close()


@app.get("/compare")
def compare():
    session = get_session()
    try:
        rows = session.query(NlpResult).all()
        return [{c.name: getattr(r, c.name) for c in NlpResult.__table__.columns} for r in rows]
    finally:
        session.close()


@app.post("/search")
def search(req: SearchRequest):
    store = TranscriptStore()
    return store.search(req.query, n_results=req.n_results)


@app.get("/correlations")
def correlations():
    df = run_correlations()
    return df.to_dict(orient="records")
