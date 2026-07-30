# earnings-lens

NLP pipeline for NSE earnings-call transcripts: sentiment, topics, entities,
tone (hedging/certainty/evasiveness/readability), forward-guidance
extraction, and correlation against financial outcomes. Streamlit
dashboard + FastAPI backend on top.

**This is a portfolio/demo project running entirely on synthetic data.**
See [Limitations](#limitations) before reading anything here as a real
market finding.

## Quickstart

```bash
pip install -r requirements.txt
# optional, for real NER instead of the regex-only fallback:
python -m spacy download en_core_web_sm

python data/generate_sample_transcripts.py   # writes data/processed/{transcripts.json,financial_outcomes.csv}
python -m pipeline.runner                    # populates SQLite + local Chroma store
python -m analysis.correlations               # prints Spearman feature/outcome table

uvicorn api:app --reload                      # FastAPI backend
streamlit run app.py                          # dashboard
pytest -q                                     # test suite
```

## Architecture

```
data/
  scrapers/{bse_scraper,screener_scraper}.py   # BeautifulSoup parsers, fixture-driven, not live-invoked
  downloaders/pdf_downloader.py                # cache-by-ticker/quarter, live download path unused by default
  generate_sample_transcripts.py               # synthetic 5-company x 6-quarter corpus generator
  raw/, processed/                             # generated fixtures / synthetic outputs (gitignored contents)
pipeline/
  parser.py       # pdfplumber PDF extraction + section/speaker detection, also parses the synthetic text format
  sentiment.py    # FinBERT (optional) -> finance lexicon fallback
  topics.py       # BERTopic (optional) -> TF-IDF+KMeans over a fixed seed-topic vocabulary
  ner.py          # spaCy trf/sm (optional) -> regex-only fallback; regulator + numeric-claim patterns always run
  tone.py         # hedging/certainty/evasiveness (TF-IDF cosine Q&A similarity)/readability/verbosity/forward-tense
  guidance.py     # regex/spaCy guidance extraction; Claude-API structuring is a stub, never called
  analyst_qa.py   # question sentiment/topics + answer-quality via tone.py similarity
  runner.py       # orchestrates the above over the synthetic corpus, writes to SQLite + Chroma
analysis/
  feature_builder.py  # joins nlp_results + financial_outcomes into one table
  correlations.py     # Spearman rho + p-value per (feature, outcome) pair
store/
  models.py, db.py    # SQLAlchemy, SQLite by default (swap DATABASE_URL for Postgres, no model changes needed)
  chroma_store.py      # local persistent Chroma client; sentence-transformers (optional) -> TF-IDF fallback
api.py, app.py          # FastAPI backend, Streamlit dashboard
notebooks/               # eda.ipynb, correlation_analysis.ipynb -- executed, real output cells
tests/
```

## Methodology

1. **Data**: 30 synthetic transcripts (5 fictional NSE-style companies x 6
   quarters), template-generated with a per-quarter hidden "tone" value that
   biases both the language (confident vs cautious word banks) and a
   correlated-but-noisy synthetic financial outcome. This gives the
   correlation engine a real, partially-recoverable signal to find, rather
   than pure noise or a fabricated one.
2. **Parsing**: transcripts are diarized into `[MANAGEMENT]`/`[ANALYST]`
   turns; questions are paired with the next management turn; sentences
   containing guidance keywords (`expect`, `guidance`, `outlook`, `guide`,
   `forecast`) are pulled out as guidance statements.
3. **NLP**: each module tries a "real" heavy model first (FinBERT,
   BERTopic, spaCy `en_core_web_trf`, sentence-transformers) and falls back
   to a lightweight substitute if the import/load fails for any reason
   (not installed, no network, offline). The pipeline runner defaults
   `use_finbert=False` explicitly, since `transformers` being importable
   does **not** mean the ~400MB FinBERT weights are cached locally --
   defaulting to True would silently trigger a multi-hundred-MB download
   per thread the first time the pipeline runs (this happened during
   development and was fixed; see the `use_finbert` flag on
   `pipeline/runner.py`).
4. **Storage**: SQLite (`data/processed/earnings_lens.db`) via SQLAlchemy;
   swapping `EARNINGS_LENS_DB_URL` to a Postgres URL requires no model
   changes. Chroma persists locally to `data/processed/chroma/`.
5. **Correlation**: Spearman rank correlation + p-value for every
   (NLP feature) x (financial outcome) pair, run once and reported in
   full below -- including non-significant results.

## Actual correlation findings (from a real run of `analysis/correlations.py` on the 30-transcript synthetic corpus)

```
                 feature                   outcome  n     rho  p_value  significant
sentiment_weighted_score revenue_growth_actual_pct 30  0.2367   0.2078        False
sentiment_weighted_score      revenue_surprise_pct 30  0.3710   0.0436         True
sentiment_weighted_score            return_30d_pct 30  0.1967   0.2974        False
sentiment_weighted_score              guidance_met 30  0.1928   0.3073        False
     sentiment_pos_ratio revenue_growth_actual_pct 30  0.2070   0.2724        False
     sentiment_pos_ratio      revenue_surprise_pct 30  0.4163   0.0221         True
     sentiment_pos_ratio            return_30d_pct 30  0.1308   0.4910        False
     sentiment_pos_ratio              guidance_met 30  0.3328   0.0724        False
     sentiment_neg_ratio revenue_growth_actual_pct 30 -0.3723   0.0428         True
     sentiment_neg_ratio      revenue_surprise_pct 30 -0.2031   0.2818        False
     sentiment_neg_ratio            return_30d_pct 30 -0.0760   0.6898        False
     sentiment_neg_ratio              guidance_met 30 -0.0895   0.6380        False
     mgmt_vs_analyst_gap revenue_growth_actual_pct 30  0.1137   0.5496        False
     mgmt_vs_analyst_gap      revenue_surprise_pct 30  0.1531   0.4192        False
     mgmt_vs_analyst_gap            return_30d_pct 30 -0.0461   0.8090        False
     mgmt_vs_analyst_gap              guidance_met 30  0.0578   0.7617        False
           hedging_score revenue_growth_actual_pct 30 -0.3023   0.1045        False
           hedging_score      revenue_surprise_pct 30 -0.1991   0.2916        False
           hedging_score            return_30d_pct 30  0.3489   0.0588        False
           hedging_score              guidance_met 30 -0.1165   0.5397        False
         certainty_score revenue_growth_actual_pct 30  0.0950   0.6177        False
         certainty_score      revenue_surprise_pct 30  0.2850   0.1269        False
         certainty_score            return_30d_pct 30 -0.0600   0.7527        False
         certainty_score              guidance_met 30  0.2435   0.1947        False
   forward_looking_ratio revenue_growth_actual_pct 30  0.3462   0.0609        False
   forward_looking_ratio      revenue_surprise_pct 30  0.1293   0.4960        False
   forward_looking_ratio            return_30d_pct 30 -0.2352   0.2108        False
   forward_looking_ratio              guidance_met 30  0.1986   0.2927        False
    flesch_kincaid_grade revenue_growth_actual_pct 30 -0.1360   0.4737        False
    flesch_kincaid_grade      revenue_surprise_pct 30  0.0662   0.7281        False
    flesch_kincaid_grade            return_30d_pct 30  0.1531   0.4192        False
    flesch_kincaid_grade              guidance_met 30 -0.1657   0.3817        False
         avg_evasiveness revenue_growth_actual_pct 30 -0.0597   0.7541        False
         avg_evasiveness      revenue_surprise_pct 30 -0.1268   0.5043        False
         avg_evasiveness            return_30d_pct 30  0.1044   0.5830        False
         avg_evasiveness              guidance_met 30 -0.2582   0.1683        False
       guidance_accuracy revenue_growth_actual_pct 27  0.5245   0.0050         True
       guidance_accuracy      revenue_surprise_pct 27  0.1384   0.4913        False
       guidance_accuracy            return_30d_pct 27  0.0398   0.8438        False
       guidance_accuracy              guidance_met 27  0.2750   0.1650        False

4 of 40 feature-outcome pairs significant at alpha=0.05
```

**Read honestly**: with n=30 and alpha=0.05, a handful of pairs crossing
significance by chance is expected even under a true null. The strongest
result -- `guidance_accuracy` vs `revenue_growth_actual_pct` (rho=0.52,
p=0.005) -- is somewhat mechanical: the synthetic generator biases both
guidance language and actual outcomes from the same underlying "tone"
value, so guidance direction and realised growth direction are correlated
by construction. This demonstrates the analysis pipeline works end-to-end
and produces real (not fabricated) statistics -- it is not evidence about
real markets.

## Integrations

`store/featurestore_sync.py` exports the NLP feature table
(`analysis.feature_builder.build_feature_table`) into the sibling
[lite-featurestore](https://github.com/siddharthgaur1/featurestore) project
as a registered `earnings_call_nlp_features` FeatureGroup (entity_id =
ticker, event_timestamp = quarter-end date). This turns per-transcript
signals (sentiment, hedging, evasiveness, guidance accuracy, ...) into
something a downstream model (e.g. nifty-forecaster) can pull point-in-time
correct via `FeatureStore.get_historical_features`, instead of re-running
this whole NLP pipeline itself. Optional dependency — `pip install -e
../featurestore` (not published to PyPI) to use it:

```python
from featurestore import FeatureStore
from store.featurestore_sync import sync_to_feature_store

fs = FeatureStore(config="featurestore.yaml")
sync_to_feature_store(fs)  # idempotent, safe to call after every pipeline run
```

## Limitations

- **Synthetic data by default.** All 30 transcripts and their financial
  outcomes are template-generated (`data/generate_sample_transcripts.py`),
  not scraped from real filings. Clearly synthetic, not fabricated-as-real.
- **No live scraping.** `data/scrapers/bse_scraper.py` and
  `screener_scraper.py` are structurally complete (real BeautifulSoup
  parsing logic, documented fixture HTML format) but are never invoked
  against bseindia.com / screener.in / NSE / Tijori anywhere in this repo
  -- ToS and reliability risk, out of scope for a demo.
- **Fallback NLP models by default.** FinBERT, BERTopic, spaCy
  `en_core_web_trf`, and sentence-transformer embeddings are fully coded
  as the "real" path per module and are used automatically if the
  dependency is importable *and* loads successfully -- but the pipeline
  runner defaults to the lightweight fallbacks (finance lexicon, TF-IDF +
  KMeans, spaCy `en_core_web_sm` / regex, TF-IDF vectors) so a fresh
  `pip install -r requirements.txt` runs the whole thing offline without
  multi-hundred-MB-to-multi-GB downloads.
- **No live Claude API calls.** `pipeline/guidance.py`'s
  `structure_guidance_with_claude` is a stub that raises
  `NotImplementedError` and is not called anywhere in the pipeline, tests,
  or demos. Guidance extraction is regex/spaCy rule-based only.
- **No live yfinance calls.** Financial outcomes are synthetic
  (correlated-with-noise to the transcript's hidden tone, not real stock
  data), guaranteeing the demo runs fully offline.
- **Evasiveness/answer-quality is lexical, not semantic.** Q&A similarity
  uses TF-IDF cosine similarity, not transformer embeddings -- a real
  upgrade path noted directly in `pipeline/tone.py`.
- **SQLite, not Postgres; local Chroma, not a hosted service.** Both are
  drop-in swappable (`EARNINGS_LENS_DB_URL` env var; Chroma
  `PersistentClient`) with no model/schema changes.

## Testing

```
pytest -q
```
18 tests covering the parser (diarization, section/guidance detection,
untagged-text fallback), the sentiment lexicon fallback (including
negation handling and the management-vs-analyst gap flag), the tone
module's evasiveness detector (on-topic vs evasive answers, hedging,
forward-looking ratio), the correlation math (row shape, perfect
correlation significance, small-n graceful handling), and the Chroma
store's index/search roundtrip including the cross-process TF-IDF
re-fit path.
