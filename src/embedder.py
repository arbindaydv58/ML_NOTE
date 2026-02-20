from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

def create_embedder():
    return SentenceTransformer(EMBED_MODEL)

def embed(model, texts):
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")