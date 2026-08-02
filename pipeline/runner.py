"""Pipeline runner.

    python pipeline/runner.py --ticker TCS.NS --quarters 8
    python pipeline/runner.py --all

Mode auto-detection: live BSE/Screener/NSE scraping is not wired into this
runner (see data/scrapers/*.py docstrings for why). The runner always runs
against data/processed/transcripts.json, the synthetic corpus produced by
data/generate_sample_transcripts.py -- it logs this clearly ("SYNTHETIC
MODE") rather than pretending otherwise. If that file is missing it is
generated on the fly.

Incremental: rows already present in nlp_results for a given
ticker/quarter/year are skipped unless --force is passed.
Concurrency: NLP stages for different transcripts run in a
ThreadPoolExecutor (they're independent; mostly CPU/regex-bound with small
models, so a thread pool is enough -- no need for multiprocessing here).
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.guidance import extract_all_guidance, guidance_accuracy
from pipeline.ner import entity_frequency, extract_entities
from pipeline.parser import parse_synthetic_record
from pipeline.sentiment import management_vs_analyst_gap, score_sentences
from pipeline.tone import (
    analyze_qa_tone,
    certainty_score,
    flesch_kincaid_grade,
    forward_looking_ratio,
    hedging_score,
)
from pipeline.topics import top_topics_for_text
from store.chroma_store import TranscriptStore
from store.db import get_session, init_db
from store.models import FinancialOutcome, NlpResult, Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("runner")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TRANSCRIPTS_PATH = PROCESSED_DIR / "transcripts.json"
OUTCOMES_PATH = PROCESSED_DIR / "financial_outcomes.csv"


def _ensure_synthetic_data():
    if not TRANSCRIPTS_PATH.exists() or not OUTCOMES_PATH.exists():
        log.info("SYNTHETIC MODE: no processed data found, generating it now.")
        from data.generate_sample_transcripts import main as generate_main

        generate_main()
    else:
        log.info("SYNTHETIC MODE: using existing data/processed/transcripts.json (live scraping not enabled).")


def load_transcripts(ticker: str | None = None, max_quarters: int | None = None) -> list[dict]:
    records = json.loads(TRANSCRIPTS_PATH.read_text(encoding="utf-8"))
    if ticker:
        records = [r for r in records if r["ticker"] == ticker]
    if max_quarters:
        records = records[:max_quarters]
    return records


def load_outcomes() -> dict:
    import csv

    outcomes = {}
    with open(OUTCOMES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["ticker"], row["quarter"], int(row["year"]))
            outcomes[key] = {
                "revenue_growth_actual_pct": float(row["revenue_growth_actual_pct"]),
                "revenue_surprise_pct": float(row["revenue_surprise_pct"]),
                "return_30d_pct": float(row["return_30d_pct"]),
                "guidance_met": row["guidance_met"] == "True",
            }
    return outcomes


def process_one(record: dict, outcome: dict | None, use_finbert: bool = False) -> dict:
    """Run the full NLP pipeline on one transcript record. Returns a dict
    ready to persist as an NlpResult row, plus the parsed transcript.
    """
    parsed = parse_synthetic_record(record)

    sentiment = score_sentences(parsed["management_text"], use_finbert=use_finbert)
    gap = management_vs_analyst_gap(parsed["management_text"], parsed["analyst_qa"], use_finbert=use_finbert)

    full_text = parsed["management_text"] + " " + " ".join(p["answer"] for p in parsed["analyst_qa"])
    topics = top_topics_for_text(full_text)

    qa_tone = analyze_qa_tone(parsed["analyst_qa"])
    avg_evasiveness = (
        round(sum(1 for t in qa_tone if t["evasive"]) / len(qa_tone), 4) if qa_tone else 0.0
    )
    n_evasive = sum(1 for t in qa_tone if t["evasive"])

    guidance_items = extract_all_guidance(parsed["guidance_statements"])
    accuracy = guidance_accuracy(guidance_items, outcome or {})

    entities = extract_entities(parsed["management_text"])
    ent_freq = entity_frequency(entities)

    nlp_row = {
        "ticker": parsed["ticker"],
        "quarter": parsed["quarter"],
        "year": parsed["year"],
        "sentiment_engine": sentiment.engine,
        "sentiment_pos_ratio": sentiment.pos_ratio,
        "sentiment_neu_ratio": sentiment.neu_ratio,
        "sentiment_neg_ratio": sentiment.neg_ratio,
        "sentiment_weighted_score": sentiment.weighted_score,
        "mgmt_vs_analyst_gap": gap["gap"],
        "mgmt_vs_analyst_flag": gap["flag"],
        "topics_engine": "seed_keyword_fallback",
        "top_topics": topics,
        "hedging_score": hedging_score(parsed["management_text"]),
        "certainty_score": certainty_score(parsed["management_text"]),
        "forward_looking_ratio": forward_looking_ratio(parsed["management_text"]),
        "flesch_kincaid_grade": flesch_kincaid_grade(parsed["management_text"]),
        "avg_evasiveness": avg_evasiveness,
        "n_evasive_answers": n_evasive,
        "guidance_items": guidance_items,
        "guidance_accuracy": accuracy.get("accuracy"),
        "entity_frequency": ent_freq,
    }
    return {"parsed": parsed, "nlp_row": nlp_row}


def run_pipeline(ticker: str | None = None, quarters: int | None = None, force: bool = False, use_finbert: bool = False, max_workers: int = 4) -> dict:
    _ensure_synthetic_data()
    init_db()

    records = load_transcripts(ticker=ticker, max_quarters=quarters)
    outcomes = load_outcomes()

    session = get_session()
    try:
        existing_keys = {(r.ticker, r.quarter, r.year) for r in session.query(NlpResult.ticker, NlpResult.quarter, NlpResult.year)}
    finally:
        session.close()

    todo = []
    skipped = 0
    for r in records:
        key = (r["ticker"], r["quarter"], r["year"])
        if not force and key in existing_keys:
            skipped += 1
            continue
        todo.append(r)

    log.info(f"{len(records)} transcripts total, {skipped} already processed (skipped), {len(todo)} to run.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for r in todo:
            key = (r["ticker"], r["quarter"], r["year"])
            futures[executor.submit(process_one, r, outcomes.get(key), use_finbert)] = r
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            log.info(f"[{done}/{len(todo)}] processed {futures[future]['ticker']} {futures[future]['quarter']}{futures[future]['year']}")

    # persist
    session = get_session()
    chroma_docs = []
    try:
        for item in results:
            parsed, nlp_row = item["parsed"], item["nlp_row"]
            transcript = Transcript(
                ticker=parsed["ticker"],
                company=parsed["company"],
                sector=next((rec.get("sector") for rec in records if rec["ticker"] == parsed["ticker"]), None),
                quarter=parsed["quarter"],
                year=parsed["year"],
                management_text=parsed["management_text"],
                analyst_qa=parsed["analyst_qa"],
                guidance_statements=parsed["guidance_statements"],
                raw_text=parsed["management_text"],
            )
            session.merge(transcript)
            session.merge(NlpResult(**nlp_row))
            chroma_docs.append(
                {
                    "id": f"{parsed['ticker']}_{parsed['quarter']}{parsed['year']}",
                    "text": parsed["management_text"],
                    "metadata": {"ticker": parsed["ticker"], "quarter": parsed["quarter"], "year": parsed["year"]},
                }
            )
        for key, outcome in outcomes.items():
            t, q, y = key
            session.merge(FinancialOutcome(ticker=t, quarter=q, year=y, **outcome))
        session.commit()
    finally:
        session.close()

    chroma_stats = {"engine": "n/a", "n_indexed": 0}
    if chroma_docs:
        store = TranscriptStore()
        chroma_stats = store.index_transcripts(chroma_docs)

    return {
        "n_total": len(records),
        "n_skipped": skipped,
        "n_processed": len(todo),
        "chroma": chroma_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="earnings-lens pipeline runner")
    parser.add_argument("--ticker", type=str, default=None, help="e.g. TCS.NS; omit to run all companies")
    parser.add_argument("--quarters", type=int, default=None, help="max quarters per ticker")
    parser.add_argument("--force", action="store_true", help="reprocess even if already in DB")
    parser.add_argument("--use-finbert", action="store_true", help="use FinBERT if installed (may trigger a model download)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    summary = run_pipeline(ticker=args.ticker, quarters=args.quarters, force=args.force, use_finbert=args.use_finbert, max_workers=args.workers)
    log.info(f"Done. {summary}")


if __name__ == "__main__":
    main()
