from pipeline.parser import parse_text, extract_guidance_sentences, diarize

SAMPLE = """[MANAGEMENT] We had a strong quarter with solid revenue growth.
[MANAGEMENT] We expect margins to improve next year.

[ANALYST] What is your outlook on capex?
[MANAGEMENT] We plan to increase capex to support demand.
"""


def test_diarize_splits_by_tag():
    turns = diarize(SAMPLE)
    speakers = [s for s, _ in turns]
    assert speakers == ["MANAGEMENT", "MANAGEMENT", "ANALYST", "MANAGEMENT"]


def test_parse_text_extracts_management_and_qa():
    result = parse_text(SAMPLE, company="Acme", ticker="ACME.NS", quarter="Q1", year=2024)
    assert result["company"] == "Acme"
    assert "strong quarter" in result["management_text"]
    assert len(result["analyst_qa"]) == 1
    assert result["analyst_qa"][0]["question"] == "What is your outlook on capex?"
    assert "capex" in result["analyst_qa"][0]["answer"]


def test_guidance_sentence_detection():
    text = "We had a good quarter. We expect growth of 10% next year. Weather was nice."
    sentences = extract_guidance_sentences(text)
    assert len(sentences) == 1
    assert "expect" in sentences[0].lower()


def test_parse_text_untagged_falls_back_to_whole_text_as_management():
    result = parse_text("Just some plain extracted PDF text with no speaker tags.")
    assert result["management_text"]
    assert result["analyst_qa"] == []
