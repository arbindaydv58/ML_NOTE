import os
import pickle
import faiss

from sentence_transformers import CrossEncoder
from groq import Groq

from .config import *
from .loader import load_slides
from .embedder import create_embedder, embed
from .search import build_index


DB_INDEX = "database/faiss.index"
DB_META = "database/metadata.pkl"


print("Loading embedding model...")
embed_model = create_embedder()


def build_and_save():

    print("Loading slides from PPT files...")
    documents, metadata = load_slides()

    if len(documents) == 0:
        raise Exception("NO SLIDES FOUND — check your data/week folders")

    print(f"Slides loaded: {len(documents)}")

    print("Embedding slides...")
    embeddings = embed(embed_model, documents)

    print("Building FAISS index...")
    index = build_index(embeddings)

    print("Saving database...")

    os.makedirs("database", exist_ok=True)

    faiss.write_index(index, DB_INDEX)

    with open(DB_META, "wb") as f:
        pickle.dump((documents, metadata), f)

    print("Database saved.")

    return index, documents, metadata


# SAFE LOAD OR BUILD

if (
    os.path.exists(DB_INDEX)
    and os.path.exists(DB_META)
    and os.path.getsize(DB_INDEX) > 1000
):

    try:
        print("Loading saved FAISS database...")

        index = faiss.read_index(DB_INDEX)

        with open(DB_META, "rb") as f:
            documents, metadata = pickle.load(f)

    except:
        print("Database corrupted → rebuilding")
        index, documents, metadata = build_and_save()

else:

    print("No valid database found → building new one")
    index, documents, metadata = build_and_save()


print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

print("Connecting to Groq...")
client = Groq(api_key=GROQ_API_KEY)


def ask(question):

    q_emb = embed(embed_model, [question])

    D, I = index.search(q_emb, TOP_K)

    retrieved = [documents[i] for i in I[0]]

    pairs = [[question, doc] for doc in retrieved]

    # DISABLE batch progress bar
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(retrieved, scores),
        key=lambda x: x[1],
        reverse=True
    )

    best = [r[0] for r in ranked[:FINAL_K]]

    context = "\n\n".join(best)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content":
                "Answer from lecture notes first. "
                "If notes do not contain the answer, use general knowledge."
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION:{question}"
            }
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content