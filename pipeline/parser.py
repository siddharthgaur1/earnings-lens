"""Transcript parsing: PDF text extraction + section/speaker detection.

Two entry points:
- `parse_pdf(path)` -- real pdfplumber-based extraction from a PDF file,
  then runs the same text-parsing logic below.
- `parse_text(raw_text, ...)` -- parses an already-extracted transcript
  string (what the synthetic generator, or `parse_pdf`, produces). This is
  what the runner uses against the synthetic corpus.

Output shape (both paths converge here)::

    {
        "company": str, "ticker": str, "quarter": str, "year": int,
        "management_text": str,
        "analyst_qa": [{"question": str, "answer": str}, ...],
        "guidance_statements": [str, ...],
    }
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

GUIDANCE_KEYWORDS = re.compile(r"\b(expect|guidance|outlook|guide[ds]?|forecast)\b", re.IGNORECASE)

SPEAKER_TAG_RE = re.compile(r"^\[(MANAGEMENT|ANALYST)\]\s*(.*)$")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_text_from_pdf(path: Path) -> str:
    """Real pdfplumber extraction path."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_guidance_sentences(text: str) -> list[str]:
    """Pull out sentences that look like forward guidance."""
    return [s for s in _split_sentences(text) if GUIDANCE_KEYWORDS.search(s)]


def _diarize_tagged(raw_text: str) -> list[tuple[str, str]]:
    """Split [MANAGEMENT]/[ANALYST]-tagged text into (speaker, line) pairs."""
    turns = []
    for line in raw_text.splitlines():
        m = SPEAKER_TAG_RE.match(line.strip())
        if m:
            turns.append((m.group(1), m.group(2)))
    return turns


def _diarize_heuristic(raw_text: str) -> list[tuple[str, str]]:
    """Fallback speaker diarization for untagged text (e.g. real PDF
    extraction that doesn't carry [MANAGEMENT]/[ANALYST] markers): look for
    common "Name:" / "Analyst:" / "Q:" / "A:" style prefixes line-by-line.
    """
    turns = []
    analyst_prefix = re.compile(r"^(Q|Analyst|Question)\s*[:\-]", re.IGNORECASE)
    mgmt_prefix = re.compile(r"^(A|Management|CEO|CFO|MD)\s*[:\-]", re.IGNORECASE)
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if analyst_prefix.match(line):
            turns.append(("ANALYST", analyst_prefix.sub("", line).strip()))
        elif mgmt_prefix.match(line):
            turns.append(("MANAGEMENT", mgmt_prefix.sub("", line).strip()))
        else:
            # continuation of the previous speaker, or untagged prose -> treat as management commentary
            turns.append(("MANAGEMENT", line))
    return turns


def diarize(raw_text: str) -> list[tuple[str, str]]:
    tagged = _diarize_tagged(raw_text)
    if tagged:
        return tagged
    return _diarize_heuristic(raw_text)


def _pair_qa(turns: list[tuple[str, str]]) -> list[dict]:
    """Turn a flat speaker-tagged sequence into question/answer pairs.
    A question is any ANALYST turn; its answer is the next MANAGEMENT turn.
    """
    pairs = []
    pending_question = None
    for speaker, line in turns:
        if speaker == "ANALYST":
            pending_question = line
        elif speaker == "MANAGEMENT" and pending_question is not None:
            pairs.append({"question": pending_question, "answer": line})
            pending_question = None
    return pairs


def parse_text(
    raw_text: str,
    company: str = "",
    ticker: str = "",
    quarter: str = "",
    year: Optional[int] = None,
) -> dict:
    turns = diarize(raw_text)

    # everything before the first ANALYST turn (opening remarks / prepared
    # commentary) is management_text; if the text has no diarization tags
    # at all, the whole thing is management_text.
    mgmt_lines = []
    first_analyst_idx = next((i for i, (s, _) in enumerate(turns) if s == "ANALYST"), None)
    if first_analyst_idx is None:
        mgmt_lines = [l for s, l in turns if s == "MANAGEMENT"]
    else:
        mgmt_lines = [l for s, l in turns[:first_analyst_idx] if s == "MANAGEMENT"]
    management_text = " ".join(mgmt_lines).strip()
    if not management_text and not turns:
        management_text = raw_text.strip()

    analyst_qa = _pair_qa(turns)
    guidance_statements = extract_guidance_sentences(management_text)
    # also scan management answers within Q&A for guidance-flavoured statements
    for pair in analyst_qa:
        guidance_statements.extend(extract_guidance_sentences(pair["answer"]))

    return {
        "company": company,
        "ticker": ticker,
        "quarter": quarter,
        "year": year,
        "management_text": management_text,
        "analyst_qa": analyst_qa,
        "guidance_statements": guidance_statements,
    }


def parse_pdf(path: Path, company: str = "", ticker: str = "", quarter: str = "", year: Optional[int] = None) -> dict:
    raw_text = extract_text_from_pdf(path)
    return parse_text(raw_text, company=company, ticker=ticker, quarter=quarter, year=year)


def parse_synthetic_record(record: dict) -> dict:
    """Parse the structured dict emitted directly by
    data/generate_sample_transcripts.py (has raw_text plus known
    company/ticker/quarter/year already) -- the fast path the demo uses.
    """
    return parse_text(
        record["raw_text"],
        company=record.get("company", ""),
        ticker=record.get("ticker", ""),
        quarter=record.get("quarter", ""),
        year=record.get("year"),
    )
