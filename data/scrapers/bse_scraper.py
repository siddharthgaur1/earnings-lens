"""BSE (bseindia.com) earnings-call transcript scraper.

NOT invoked against the live site by default anywhere in this repo (tests,
runner, demos). It is structured to parse the transcript-listing and
transcript-detail HTML that bseindia.com serves, against a *local* HTML
fixture, so the interface is real and testable without hitting the network
or risking BSE's ToS.

To point this at the live site, pass a real URL to `fetch_listing_page`
(uses `requests`, not included in default demo flow) instead of
`fetch_listing_from_fixture`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup


def fetch_listing_from_fixture(fixture_path: Path) -> str:
    """Read a locally saved BSE listing-page HTML fixture (documented format
    below) instead of making a network call.
    """
    return Path(fixture_path).read_text(encoding="utf-8")


def fetch_listing_page(url: str, session=None) -> str:  # pragma: no cover - live path, unused by default
    """Live network path. Not called by any test/demo in this repo.

    Expects `requests` to be installed by the caller; imported lazily so it
    is not a hard dependency of this module.
    """
    import requests

    sess = session or requests.Session()
    resp = sess.get(url, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_transcript_links(html: str) -> list[dict]:
    """Parse a BSE transcript-listing page.

    Documented fixture format (what `fetch_listing_from_fixture` should
    return): a page with rows like::

        <table id="transcript-list">
          <tr class="transcript-row" data-ticker="TCS" data-quarter="Q1"
              data-year="2024">
            <td><a href="/transcripts/tcs_q1_2024.pdf">Download</a></td>
          </tr>
        </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.transcript-row")
    results = []
    for row in rows:
        link = row.find("a")
        if not link or not link.get("href"):
            continue
        results.append(
            {
                "ticker": row.get("data-ticker"),
                "quarter": row.get("data-quarter"),
                "year": row.get("data-year"),
                "pdf_url": link["href"],
            }
        )
    return results
