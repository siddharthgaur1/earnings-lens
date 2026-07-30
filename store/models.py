"""SQLAlchemy models: transcripts, nlp_results, financial_outcomes.

Defaults to SQLite for local/demo runs. The connection string
(store/db.py: DATABASE_URL) is a plain SQLAlchemy URL, so swapping to
Postgres later is just changing that one string (e.g.
"postgresql://user:pass@host/dbname") -- no model changes required.
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, String, Float, Text, Boolean, JSON, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    company = Column(String, nullable=False)
    sector = Column(String)
    quarter = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    management_text = Column(Text)
    analyst_qa = Column(JSON)
    guidance_statements = Column(JSON)
    raw_text = Column(Text)

    __table_args__ = (UniqueConstraint("ticker", "quarter", "year", name="uq_transcript_ticker_quarter_year"),)


class NlpResult(Base):
    __tablename__ = "nlp_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcript_id = Column(Integer, nullable=True)  # informational only; the real join key is (ticker, quarter, year)
    ticker = Column(String, nullable=False)
    quarter = Column(String, nullable=False)
    year = Column(Integer, nullable=False)

    sentiment_engine = Column(String)
    sentiment_pos_ratio = Column(Float)
    sentiment_neu_ratio = Column(Float)
    sentiment_neg_ratio = Column(Float)
    sentiment_weighted_score = Column(Float)
    mgmt_vs_analyst_gap = Column(Float)
    mgmt_vs_analyst_flag = Column(Boolean)

    topics_engine = Column(String)
    top_topics = Column(JSON)

    hedging_score = Column(Float)
    certainty_score = Column(Float)
    forward_looking_ratio = Column(Float)
    flesch_kincaid_grade = Column(Float)
    avg_evasiveness = Column(Float)
    n_evasive_answers = Column(Integer)

    guidance_items = Column(JSON)
    guidance_accuracy = Column(Float)

    entity_frequency = Column(JSON)

    __table_args__ = (UniqueConstraint("ticker", "quarter", "year", name="uq_nlp_ticker_quarter_year"),)


class FinancialOutcome(Base):
    __tablename__ = "financial_outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    quarter = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    revenue_growth_actual_pct = Column(Float)
    revenue_surprise_pct = Column(Float)
    return_30d_pct = Column(Float)
    guidance_met = Column(Boolean)

    __table_args__ = (UniqueConstraint("ticker", "quarter", "year", name="uq_outcome_ticker_quarter_year"),)
