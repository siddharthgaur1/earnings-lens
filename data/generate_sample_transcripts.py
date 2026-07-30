"""Synthetic earnings-call transcript + financial-outcome generator.

SYNTHETIC DATA ONLY. This does not scrape or reproduce any real transcript.
It template-generates plausible-looking management commentary, analyst Q&A,
and forward guidance for 5 fictionalised NSE-style companies across 6
quarters (30 transcripts total), plus a matching synthetic financial
outcomes table. Output goes to data/processed/transcripts.json and
data/processed/financial_outcomes.csv.

The "tone" of each transcript (a hidden -1..1 mood value per quarter) is
sampled per company/quarter and used to bias word-bank choices (confident vs
hedgy language) AND to bias the synthetic actual financial outcome, with
noise on top. That gives the downstream correlation engine a real, partially
correlated (not perfectly, not cherry-picked) signal to discover.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

COMPANIES = [
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT Services"},
    {"ticker": "INFY.NS", "name": "Infosys", "sector": "IT Services"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Conglomerate"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive"},
]

QUARTERS = [("Q1", 2024), ("Q2", 2024), ("Q3", 2024), ("Q4", 2024), ("Q1", 2025), ("Q2", 2025)]

CONFIDENT_OPENERS = [
    "We delivered a strong quarter with broad-based growth across segments.",
    "This has been one of our best quarters in recent years, and momentum is clearly building.",
    "I am pleased to report robust performance, well ahead of our internal expectations.",
    "Our results this quarter reflect the strength of our diversified portfolio.",
]
CAUTIOUS_OPENERS = [
    "This was a challenging quarter, with headwinds in a few key segments.",
    "We navigated a difficult macro environment, and performance was largely in line with a soft market.",
    "Results this quarter were impacted by weaker demand and currency volatility.",
    "It has been a mixed quarter, and we are watching the environment closely.",
]

CONFIDENT_BODY = [
    "Revenue growth was driven by strong deal wins and healthy client additions.",
    "Margins expanded on the back of operating leverage and disciplined cost control.",
    "We continue to see strong demand in our core segments and are scaling capacity accordingly.",
    "Our order book remains healthy and gives us good visibility into the coming quarters.",
    "Attrition has stabilised and we are confident in our talent pipeline.",
]
CAUTIOUS_BODY = [
    "We expect revenue growth to remain muted given the uncertain demand environment.",
    "Margins were under pressure due to elevated input costs and wage inflation.",
    "Client budgets have been tighter than anticipated, and deal cycles have lengthened.",
    "We are cautious about the near-term outlook given continued macro uncertainty.",
    "Attrition remains a challenge and we are investing in retention.",
]

HEDGE_PHRASES_HIGH = [
    "It is possible that this trend could potentially continue, although it may somewhat depend on market conditions.",
    "We think it might perhaps be reasonable to expect some improvement, but this is not certain.",
    "There could be some upside, though it is difficult to say with confidence at this stage.",
]
HEDGE_PHRASES_LOW = [
    "We are confident this trend will continue.",
    "We expect a clear improvement next quarter.",
    "This is a definite positive for the business going forward.",
]

GUIDANCE_TEMPLATES_UP = [
    "For the coming year, we expect revenue growth of {lo}-{hi}%, an increase from our prior guidance.",
    "We are raising our full-year margin guidance to {lo}-{hi}%, reflecting continued operating efficiency.",
    "Looking ahead, we guide for EBITDA growth in the range of {lo}-{hi}% over the next fiscal year.",
]
GUIDANCE_TEMPLATES_DOWN = [
    "We are revising our revenue growth guidance down to {lo}-{hi}% for the full year.",
    "Given the current environment, we expect margins to be in the range of {lo}-{hi}%, lower than earlier guided.",
    "We now expect full-year growth of only {lo}-{hi}%, below our previous outlook.",
]
GUIDANCE_TEMPLATES_FLAT = [
    "We expect growth to be broadly in line with this year, in the {lo}-{hi}% range.",
    "Our outlook for margins remains steady at {lo}-{hi}% for the next few quarters.",
]

ANALYST_QUESTIONS = [
    "Can you help us understand the drivers behind this quarter's margin performance?",
    "What is your outlook on revenue growth for the next two quarters?",
    "How should we think about capex plans given the current demand environment?",
    "Could you comment on competitive intensity in your core markets?",
    "What's driving the change in your guidance versus last quarter?",
    "Can you give more colour on hiring plans and wage inflation?",
    "How are you thinking about debt levels and capital allocation going forward?",
    "Any update on regulatory developments from SEBI or RBI that could affect the business?",
]

ANSWER_ON_TOPIC_CONFIDENT = [
    "Sure, happy to address that. Margin improvement this quarter came from a mix of operating leverage, pricing discipline, and lower attrition-related costs, and we expect this to sustain.",
    "Growth this quarter was broad based across geographies and verticals, and we expect similar momentum to continue into the next two quarters.",
    "We plan to continue investing in capacity, and our capex plans remain unchanged, focused on high-return areas.",
]
ANSWER_ON_TOPIC_CAUTIOUS = [
    "That's a fair question. Margins were impacted by a few one-off costs this quarter, and while we expect some normalisation, visibility remains limited.",
    "Growth has been softer than we would like, driven by delayed client decision-making, and we are watching this closely before committing to a stronger outlook.",
    "We are being more conservative on capex given the uncertain demand backdrop, and will reassess as the year progresses.",
]
ANSWER_EVASIVE = [
    "That's an important area for us and something the leadership team discusses regularly as part of our broader strategic priorities.",
    "We remain focused on our long-term vision and continue to evaluate all aspects of the business holistically.",
    "There are a lot of moving parts here, and we'd rather not get into specifics on a call like this.",
]

REGULATORS = ["SEBI", "RBI", "TRAI"]
NUMERIC_CLAIM_TEMPLATES = [
    "Revenue grew {pct}% year on year to Rs {amt} crore.",
    "EBITDA margin came in at {pct2}% for the quarter.",
    "We added {n} new clients this quarter.",
]


@dataclass
class TranscriptRecord:
    company: str
    ticker: str
    sector: str
    quarter: str
    year: int
    tone: float
    management_text: str
    analyst_qa: list = field(default_factory=list)
    guidance_statements: list = field(default_factory=list)
    raw_text: str = ""


def _pick(rng: random.Random, confident_pool, cautious_pool, tone: float):
    """Weighted pick biased toward confident pool as tone increases."""
    p_confident = 0.5 + 0.4 * tone  # tone in [-1,1] -> p in [0.1, 0.9]
    p_confident = min(0.92, max(0.08, p_confident))
    pool = confident_pool if rng.random() < p_confident else cautious_pool
    return rng.choice(pool)


def _gen_management_text(rng: random.Random, tone: float) -> str:
    opener = _pick(rng, CONFIDENT_OPENERS, CAUTIOUS_OPENERS, tone)
    body_sentences = [_pick(rng, CONFIDENT_BODY, CAUTIOUS_BODY, tone) for _ in range(3)]
    hedge = _pick(rng, HEDGE_PHRASES_LOW, HEDGE_PHRASES_HIGH, tone)
    numeric = rng.choice(NUMERIC_CLAIM_TEMPLATES).format(
        pct=round(rng.uniform(-4, 18), 1),
        amt=round(rng.uniform(2000, 60000)),
        pct2=round(rng.uniform(12, 32), 1),
        n=rng.randint(3, 40),
    )
    reg = ""
    if rng.random() < 0.35:
        reg = f" We continue to remain compliant with all {rng.choice(REGULATORS)} guidelines applicable to our operations."
    return f"{opener} {' '.join(body_sentences)} {numeric} {hedge}{reg}"


def _gen_guidance(rng: random.Random, tone: float) -> list[str]:
    n = rng.randint(1, 2)
    out = []
    for _ in range(n):
        if tone > 0.15:
            tmpl = rng.choice(GUIDANCE_TEMPLATES_UP)
            lo, hi = sorted([round(rng.uniform(8, 14), 1), round(rng.uniform(14, 20), 1)])
        elif tone < -0.15:
            tmpl = rng.choice(GUIDANCE_TEMPLATES_DOWN)
            lo, hi = sorted([round(rng.uniform(0, 5), 1), round(rng.uniform(5, 9), 1)])
        else:
            tmpl = rng.choice(GUIDANCE_TEMPLATES_FLAT)
            lo, hi = sorted([round(rng.uniform(6, 9), 1), round(rng.uniform(9, 12), 1)])
        out.append(tmpl.format(lo=lo, hi=hi))
    return out


def _gen_qa(rng: random.Random, tone: float, n_pairs: int = 5) -> list[dict]:
    qs = rng.sample(ANALYST_QUESTIONS, n_pairs)
    pairs = []
    for q in qs:
        r = rng.random()
        if r < 0.15:
            a = rng.choice(ANSWER_EVASIVE)
        else:
            a = _pick(rng, ANSWER_ON_TOPIC_CONFIDENT, ANSWER_ON_TOPIC_CAUTIOUS, tone)
        pairs.append({"question": q, "answer": a})
    return pairs


def generate_transcript(rng: random.Random, company: dict, quarter: str, year: int) -> TranscriptRecord:
    tone = max(-1.0, min(1.0, rng.gauss(0.0, 0.55)))
    mgmt = _gen_management_text(rng, tone)
    guidance = _gen_guidance(rng, tone)
    qa = _gen_qa(rng, tone)

    lines = [f"Earnings Call Transcript - {company['name']} ({company['ticker']}) - {quarter} FY{year}", ""]
    lines.append("[MANAGEMENT] " + mgmt)
    lines.append("[MANAGEMENT] Forward guidance: " + " ".join(guidance))
    lines.append("")
    lines.append("Q&A Session")
    for pair in qa:
        lines.append("[ANALYST] " + pair["question"])
        lines.append("[MANAGEMENT] " + pair["answer"])
    raw_text = "\n".join(lines)

    return TranscriptRecord(
        company=company["name"],
        ticker=company["ticker"],
        sector=company["sector"],
        quarter=quarter,
        year=year,
        tone=tone,
        management_text=mgmt,
        analyst_qa=qa,
        guidance_statements=guidance,
        raw_text=raw_text,
    )


def generate_financial_outcome(rng: random.Random, rec: TranscriptRecord) -> dict:
    """Synthetic 'actual' outcome, correlated with tone plus noise (not
    perfectly) so the correlation engine has a genuine, partially-real
    signal to find rather than a fabricated one.
    """
    revenue_surprise = rec.tone * 3.0 + rng.gauss(0, 2.5)  # % surprise vs consensus
    return_30d = rec.tone * 2.0 + rng.gauss(0, 4.0)  # % stock move in 30 days post-call
    revenue_growth_actual = 8 + rec.tone * 6 + rng.gauss(0, 3)
    guidance_met = rng.random() < (0.55 + 0.25 * rec.tone)  # noisy link to tone
    return {
        "ticker": rec.ticker,
        "quarter": rec.quarter,
        "year": rec.year,
        "revenue_growth_actual_pct": round(revenue_growth_actual, 2),
        "revenue_surprise_pct": round(revenue_surprise, 2),
        "return_30d_pct": round(return_30d, 2),
        "guidance_met": guidance_met,
    }


def generate_dataset(seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    transcripts = []
    outcomes = []
    for company in COMPANIES:
        for quarter, year in QUARTERS:
            rec = generate_transcript(rng, company, quarter, year)
            transcripts.append(rec.__dict__)
            outcomes.append(generate_financial_outcome(rng, rec))
    return transcripts, outcomes


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    transcripts, outcomes = generate_dataset()

    with open(PROCESSED_DIR / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)

    import csv
    with open(PROCESSED_DIR / "financial_outcomes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(outcomes[0].keys()))
        writer.writeheader()
        writer.writerows(outcomes)

    # Also drop one raw .txt fixture per transcript, mimicking what a PDF
    # extraction step would hand off, so parser.py's text-fixture path has
    # real files to read.
    for t in transcripts:
        fname = f"{t['ticker'].replace('.', '_')}_{t['quarter']}{t['year']}.txt"
        (RAW_DIR / fname).write_text(t["raw_text"], encoding="utf-8")

    print(f"Generated {len(transcripts)} synthetic transcripts -> {PROCESSED_DIR / 'transcripts.json'}")
    print(f"Generated {len(outcomes)} synthetic financial outcomes -> {PROCESSED_DIR / 'financial_outcomes.csv'}")


if __name__ == "__main__":
    main()
