import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "database/faiss.index"
META_PATH = "database/meta.pkl"

_model = None
_index = None
_chunks = None


def _load_assets():
    global _model, _index, _chunks

    if _model is None:
        print("Loading embedding model (first time may download)...")
        _model = SentenceTransformer(MODEL_NAME)

    if _index is None or _chunks is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError("FAISS index missing. Run: python -m src.build_db")

        _index = faiss.read_index(INDEX_PATH)
        _chunks = pickle.load(open(META_PATH, "rb"))
        print("Retriever ready:", len(_chunks), "chunks")


def warmup_retriever():
    _load_assets()


def retrieve(query, k=15):
    if not query or not query.strip():
        return []

    _load_assets()

    if not _chunks:
        return []

    k = max(1, min(k, len(_chunks)))

    vec = _model.encode([query], normalize_embeddings=True)
    _, ids = _index.search(vec, k)

    return [_chunks[i] for i in ids[0] if i < len(_chunks)]
