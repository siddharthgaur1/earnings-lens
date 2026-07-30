"""Local (file-based) Chroma persistent client for transcript embeddings +
semantic search. No hosted ChromaDB service required.

Embeddings: sentence-transformers if available AND its model can be loaded
without a network fetch failure; otherwise falls back to TF-IDF vectors
(fit per-corpus, stored as the "embedding"). The TF-IDF fallback is the
default for demos in this repo -- it avoids a first-run model download so
the pipeline works fully offline; sentence-transformers is used
automatically if `prefer_sentence_transformers=True` and loading succeeds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHROMA_DIR = Path(__file__).parent.parent / "data" / "processed" / "chroma"
COLLECTION_NAME = "transcripts"

_st_model = None
_st_load_failed = False


def _get_sentence_transformer():
    global _st_model, _st_load_failed
    if _st_model is not None or _st_load_failed:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _st_load_failed = True
        _st_model = None
    return _st_model


class TfidfEmbedder:
    """Fallback embedder: fits a TF-IDF vectorizer over the corpus once and
    exposes dense vectors usable as Chroma embeddings.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=256, stop_words="english")
        self._fitted = False

    def fit(self, documents: list[str]):
        self.vectorizer.fit(documents)
        self._fitted = True
        return self

    def encode(self, documents: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(documents)
        matrix = self.vectorizer.transform(documents)
        return matrix.toarray().astype(float).tolist()


def get_chroma_client(persist_dir: Path = CHROMA_DIR):
    import chromadb

    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


class TranscriptStore:
    def __init__(self, persist_dir: Path = CHROMA_DIR, prefer_sentence_transformers: bool = False):
        self.client = get_chroma_client(persist_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.prefer_sentence_transformers = prefer_sentence_transformers
        self._tfidf = TfidfEmbedder()
        self.engine = "tfidf_fallback"

    def _embed(self, documents: list[str]) -> list[list[float]]:
        if self.prefer_sentence_transformers:
            model = _get_sentence_transformer()
            if model is not None:
                self.engine = "sentence_transformers"
                return model.encode(documents).tolist()
        self.engine = "tfidf_fallback"
        return self._tfidf.fit(documents).encode(documents)

    def index_transcripts(self, records: list[dict]):
        """records: [{"id": str, "text": str, "metadata": {...}}, ...]"""
        documents = [r["text"] for r in records]
        embeddings = self._embed(documents)
        self.collection.upsert(
            ids=[r["id"] for r in records],
            documents=documents,
            embeddings=embeddings,
            metadatas=[r["metadata"] for r in records],
        )
        return {"engine": self.engine, "n_indexed": len(records)}

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        # embed the query with whatever embedder produced the stored vectors
        if self.engine == "sentence_transformers":
            model = _get_sentence_transformer()
            query_embedding = model.encode([query]).tolist()
        else:
            if not self._tfidf._fitted:
                # fresh process/instance (e.g. a new API request): refit the
                # TF-IDF vocabulary from whatever's already indexed, so the
                # query vector has the same dimensionality as stored ones.
                existing = self.collection.get(include=["documents"])
                corpus = existing.get("documents") or [query]
                self._tfidf.fit(corpus)
            query_embedding = self._tfidf.encode([query])
        result = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, d, m, dist in zip(ids, docs, metas, dists):
            hits.append({"id": i, "text": d, "metadata": m, "distance": dist})
        return hits
