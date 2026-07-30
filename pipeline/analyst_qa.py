"""Analyst Q&A-specific analysis: question sentiment, question-topic
clustering (reuses topics.py's seed-topic fallback), and management
answer-quality (reuses tone.py's Q/A similarity as the answer-quality
proxy: higher similarity ~ more on-topic/less evasive answer).
"""
from __future__ import annotations

from pipeline.sentiment import score_sentences
from pipeline.topics import top_topics_for_text
from pipeline.tone import question_answer_similarity, is_evasive


def analyze_question(question: str) -> dict:
    sentiment = score_sentences(question, use_finbert=False)
    topics = top_topics_for_text(question, top_n=3)
    return {
        "question": question,
        "sentiment_score": sentiment.weighted_score,
        "topics": topics,
    }


def answer_quality(question: str, answer: str) -> dict:
    similarity = question_answer_similarity(question, answer)
    return {
        "similarity": similarity,
        "evasive": is_evasive(question, answer),
        "quality_score": similarity,  # higher = more directly responsive
    }


def analyze_qa_block(qa: list[dict]) -> list[dict]:
    results = []
    for pair in qa:
        q_analysis = analyze_question(pair["question"])
        quality = answer_quality(pair["question"], pair["answer"])
        results.append({**q_analysis, "answer": pair["answer"], **quality})
    return results
