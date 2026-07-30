"""PDF downloader/cache keyed by (ticker, quarter, year).

Live download (`download_pdf`) is not exercised by tests/demos in this
repo. `cache_path` and `is_cached` are the pieces the rest of the pipeline
actually relies on (pipeline/parser.py reads from the cache dir).
"""
from __future__ import annotations

from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "raw" / "pdfs"


def cache_path(ticker: str, quarter: str, year: int) -> Path:
    safe_ticker = ticker.replace(".", "_")
    return CACHE_DIR / f"{safe_ticker}_{quarter}{year}.pdf"


def is_cached(ticker: str, quarter: str, year: int) -> bool:
    return cache_path(ticker, quarter, year).exists()


def download_pdf(url: str, ticker: str, quarter: str, year: int, session=None) -> Path:  # pragma: no cover - live path
    """Download a transcript PDF and cache it. Not called by default
    anywhere in this repo -- requires a live, reachable `url`.
    """
    import requests

    dest = cache_path(ticker, quarter, year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
