import shutil
import tempfile
from pathlib import Path

from store.chroma_store import TranscriptStore


def test_index_and_search_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = TranscriptStore(persist_dir=tmp)
        docs = [
            {"id": "a", "text": "Revenue growth was strong this quarter with healthy margins.", "metadata": {"ticker": "A"}},
            {"id": "b", "text": "We faced margin pressure and weak demand this quarter.", "metadata": {"ticker": "B"}},
        ]
        stats = store.index_transcripts(docs)
        assert stats["n_indexed"] == 2
        assert stats["engine"] == "tfidf_fallback"

        hits = store.search("margin pressure", n_results=2)
        assert len(hits) == 2
        assert hits[0]["id"] in ("a", "b")

        # fresh instance (simulating a new API process) must still be able
        # to search against the already-indexed collection
        store2 = TranscriptStore(persist_dir=tmp)
        hits2 = store2.search("healthy margins", n_results=1)
        assert len(hits2) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
