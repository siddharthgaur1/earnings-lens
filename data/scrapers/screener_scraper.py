"""Screener.in company-page scraper.

Structured but NOT invoked live in this repo (tests/runner/demos use the
synthetic dataset). Parses the standard Screener.in "ratios"/"quarters"
table layout from a local HTML fixture.
"""
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


def fetch_company_page_from_fixture(fixture_path: Path) -> str:
    return Path(fixture_path).read_text(encoding="utf-8")


def fetch_company_page(url: str, session=None) -> str:  # pragma: no cover - live path, unused by default
    import requests

    sess = session or requests.Session()
    resp = sess.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_quarterly_results(html: str) -> list[dict]:
    """Parse Screener.in's quarterly-results table.

    Documented fixture format::

        <table class="data-table" id="quarters">
          <thead><tr><th>Quarter</th><th>Sales</th><th>Net Profit</th></tr></thead>
          <tbody>
            <tr><td>Q1 2024</td><td>50000</td><td>9000</td></tr>
          </tbody>
        </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="quarters")
    if table is None:
        return []
    rows = table.find("tbody").find_all("tr")
    results = []
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) >= 3:
            results.append({"quarter": cells[0], "sales": cells[1], "net_profit": cells[2]})
    return results
