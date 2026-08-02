"""Topic modelling.

Real path: BERTopic, used automatically if installed.
Fallback: TF-IDF + KMeans over a fixed seed-topic vocabulary, so topic
labels are stable and comparable across the corpus regardless of engine.
"""
from __future__ import annotations

from collections import defaultdict

SEED_TOPICS = {
    "revenue growth": ["revenue", "growth", "sales", "demand", "clients", "deal", "wins"],
    "margins": ["margin", "margins", "profitability", "cost", "efficiency"],
    "competition": ["competitive", "competition", "market share", "rival"],
    "debt": ["debt", "leverage", "borrowing", "interest"],
    "capex": ["capex", "capital expenditure", "investment", "capacity"],
    "hiring": ["hiring", "attrition", "talent", "headcount", "wage"],
    "regulation": ["sebi", "rbi", "trai", "regulatory", "compliance", "regulation"],
    "product launch": ["launch", "product", "innovation", "new offering"],
    "guidance": ["guidance", "outlook", "forecast", "expect"],
}

_TOPIC_NAMES = list(SEED_TOPICS.keys())


def _bertopic_available():
    try:
        import bertopic  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 -- optional dependency unavailable, fall back
        return False


def _seed_topic_scores(text: str) -> dict:
    """Keyword-overlap score per seed topic (used both as the fallback
    engine's core logic and to *label* whichever cluster the fallback
    KMeans assigns, since raw KMeans clusters are unlabelled).
    """
    text_l = text.lower()
    scores = {}
    for topic, keywords in SEED_TOPICS.items():
        scores[topic] = sum(text_l.count(kw) for kw in keywords)
    return scores


def top_topics_for_text(text: str, top_n: int = 5) -> list[dict]:
    """Per-transcript top-N topics + weights. Works standalone (no corpus
    needed) using seed-topic keyword matching -- this is the practical
    fallback used by analyst_qa.py's question-topic clustering too.
    """
    scores = _seed_topic_scores(text)
    total = sum(scores.values()) or 1
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"topic": t, "weight": round(s / total, 4)} for t, s in ranked if True]


def fit_corpus_topics(documents: list[str], engine_hint: bool = True) -> dict:
    """Corpus-level topic modelling. Returns per-document top-5 topics plus
    a corpus-level topic-evolution-ready table (topic x document index).

    If BERTopic is installed and `engine_hint` is True, uses it to derive
    clusters, then labels each BERTopic cluster with the nearest seed topic
    (by keyword overlap of the cluster's top words) so downstream code
    always deals with the same fixed topic vocabulary.
    """
    if engine_hint and _bertopic_available():
        return _fit_with_bertopic(documents)
    return _fit_with_tfidf_kmeans(documents)


def _label_cluster(words: list[str]) -> str:
    best_topic, best_score = "guidance", -1
    for topic, keywords in SEED_TOPICS.items():
        score = sum(1 for w in words if w in keywords)
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def _fit_with_bertopic(documents: list[str]) -> dict:  # pragma: no cover - only runs if bertopic installed
    from bertopic import BERTopic

    model = BERTopic(nr_topics=len(SEED_TOPICS))
    topics, _ = model.fit_transform(documents)
    per_doc = []
    for doc, topic_id in zip(documents, topics):
        words = [w for w, _ in model.get_topic(topic_id)] if topic_id != -1 else []
        label = _label_cluster(words) if words else "guidance"
        per_doc.append([{"topic": label, "weight": 1.0}])
    return {"engine": "bertopic", "per_document": per_doc}


def _fit_with_tfidf_kmeans(documents: list[str]) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    per_document = [top_topics_for_text(doc) for doc in documents]

    # corpus-level clustering, for a "discovered structure" cross-check /
    # topic-evolution table alongside the seed-topic keyword scores above.
    n_clusters = min(len(SEED_TOPICS), max(2, len(documents)))
    if len(documents) >= 2:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
        X = vectorizer.fit_transform(documents)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        terms = vectorizer.get_feature_names_out()
        cluster_labels = {}
        for c in range(n_clusters):
            order = km.cluster_centers_[c].argsort()[::-1][:10]
            top_words = [terms[i] for i in order]
            cluster_labels[c] = _label_cluster(top_words)
        clusters = [cluster_labels[int(l)] for l in labels]
    else:
        clusters = ["guidance"] * len(documents)

    return {"engine": "tfidf_kmeans_fallback", "per_document": per_document, "cluster_assignment": clusters}


def topic_evolution(per_document_topics: list[list[dict]], quarters: list[str]) -> dict:
    """Corpus-level topic evolution: for each seed topic, its average
    weight per quarter (in the order `quarters` is given).
    """
    evolution = defaultdict(lambda: defaultdict(list))
    for topics, quarter in zip(per_document_topics, quarters):
        weights = {t["topic"]: t["weight"] for t in topics}
        for topic in _TOPIC_NAMES:
            evolution[topic][quarter].append(weights.get(topic, 0.0))

    result = {}
    for topic, by_quarter in evolution.items():
        result[topic] = {q: round(sum(v) / len(v), 4) for q, v in by_quarter.items()}
    return result
