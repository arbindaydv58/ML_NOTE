import os, faiss, pickle
from sentence_transformers import SentenceTransformer

# ---- LOAD MODEL ONLY ONCE ----
print("Loading embedding model (first time may download)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- CHECK DATABASE ----
if not os.path.exists("database/faiss.index"):
    raise RuntimeError("FAISS index missing. Run: python -m src.build_db")

index = faiss.read_index("database/faiss.index")
chunks = pickle.load(open("database/meta.pkl","rb"))

print("Retriever ready:", len(chunks), "chunks")

def retrieve(query, k=15):

    if not query.strip():
        return []

    vec = model.encode([query], normalize_embeddings=True)
    _, ids = index.search(vec, k)

    return [chunks[i] for i in ids[0] if i < len(chunks)]