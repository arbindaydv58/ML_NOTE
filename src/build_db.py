import faiss, pickle
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_text

MODEL="all-MiniLM-L6-v2"

print("Loading embed model...")
model=SentenceTransformer(MODEL)

text=open("data/all_weeks.txt",encoding="utf-8").read()

chunks=chunk_text(text)

print("Chunks:",len(chunks))

emb=model.encode(chunks,show_progress_bar=True,normalize_embeddings=True)

index=faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

faiss.write_index(index,"database/faiss.index")

pickle.dump(chunks,open("database/meta.pkl","wb"))

print("ELITE DATABASE BUILT")