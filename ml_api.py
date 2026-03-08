import os
import pickle
import faiss
import re
import time
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.chunker import chunk_text


# ==========================================
# CONFIG
# ==========================================

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "all-MiniLM-L6-v2"

DB_INDEX = "database/faiss.index"
DB_META = "database/meta.pkl"

FALLBACK = "The question is not in the study material."

MAX_CACHE_ITEMS = 200
MAX_SESSIONS = 200

# streaming speed control
STREAM_DELAY = 0.03


# ==========================================
# FASTAPI SETUP
# =========================================

app = FastAPI(title="ELITE RAG API")

@app.on_event("startup")
def startup_event():
    try:
        load_db()
        print("FAISS database loaded")
    except Exception as e:
        print("Database loading failed:", e)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ml-note.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SYSTEM = """
You are a professional Machine Learning tutor.

STRICT RULES:
1. Answer ONLY using STUDY MATERIAL.
2. Do NOT use outside knowledge.
3. If answer not present say EXACTLY:
The question is not in the study material.
4. Never mention retrieval or chunks.
5. Keep explanation clear and student friendly.
"""


# ==========================================
# GLOBAL OBJECTS
# ==========================================

embed = SentenceTransformer(MODEL)

index = None
meta = None

cache = OrderedDict()
cache_vectors = OrderedDict()
sessions = OrderedDict()


# ==========================================
# RESET STATE
# ==========================================

def reset_runtime_state():
    cache.clear()
    cache_vectors.clear()
    sessions.clear()


def _bounded_set(store, key, value, max_items):

    if key in store:
        store.pop(key)

    store[key] = value

    while len(store) > max_items:
        store.popitem(last=False)


# ==========================================
# LOAD DATABASE
# ==========================================

def load_db():

    global index, meta

    index = faiss.read_index(DB_INDEX)
    meta = pickle.load(open(DB_META, "rb"))

    reset_runtime_state()


# ==========================================
# RETRIEVE DOCUMENTS
# ==========================================

def retrieve(q, k=None):

    if k is None:

        L = len(q.split())

        if L < 5:
            k = 3
        elif L < 12:
            k = 5
        else:
            k = 7

    vec = embed.encode([q], normalize_embeddings=True)

    _, I = index.search(vec, k)

    return [meta[i] for i in I[0]], vec


# ==========================================
# GROUNDING CHECK
# ==========================================

def is_grounded(answer, docs):

    if not answer:
        return False

    if answer.strip() == FALLBACK:
        return True

    context = " ".join(docs).lower()

    tokens = re.findall(r"[a-zA-Z0-9]{4,}", answer.lower())

    if not tokens:
        return False

    unique = set(tokens)

    hits = sum(1 for t in unique if t in context)

    ratio = hits / max(1, len(unique))

    return ratio >= 0.35


# ==========================================
# STREAM LLM TOKENS
# ==========================================

def stream_answer(prompt):

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=True
    )

    for chunk in stream:

        token = chunk.choices[0].delta.content

        if token:
            yield token


# ==========================================
# ASK STREAM ENDPOINT
# ==========================================

@app.get("/ask_stream")

def ask_stream(question: str, session: str = "default"):

    if not question.strip():
        raise HTTPException(400, "Empty question")

    # =====================================
    # semantic cache
    # =====================================

    vec = embed.encode([question], normalize_embeddings=True)

    for old_q, old_v in list(cache_vectors.items()):

        score = (vec @ old_v.T)[0][0]

        if score > 0.92:

            cache_vectors.move_to_end(old_q)
            cache.move_to_end(old_q)

            return StreamingResponse(
                iter([cache[old_q]]),
                media_type="text/plain"
            )

    # =====================================
    # retrieve docs
    # =====================================

    docs, vec = retrieve(question)

    if not docs:
        return StreamingResponse(iter([FALLBACK]), media_type="text/plain")

    if sum(len(d.split()) for d in docs) < 80:
        return StreamingResponse(iter([FALLBACK]), media_type="text/plain")

    context = "\n\n".join(docs[:4])

    history = sessions.get(session, "")

    prompt = f"""
CHAT HISTORY:
{history}

STUDY MATERIAL:
{context}

QUESTION:
{question}

Give best student-friendly explanation.
"""

    # =====================================
    # STREAM GENERATOR
    # =====================================

    def generator():

        full = ""
        buffer = ""

        for token in stream_answer(prompt):

            full += token
            buffer += token

            if len(buffer) > 25 or token.endswith((" ", ".", ",", "\n")):

                yield buffer
                buffer = ""

                # optional speed control
                time.sleep(STREAM_DELAY)

        if buffer:
            yield buffer

        if not is_grounded(full, docs):
            full = FALLBACK

        _bounded_set(cache, question, full, MAX_CACHE_ITEMS)
        _bounded_set(cache_vectors, question, vec, MAX_CACHE_ITEMS)

        updated_history = (
            history + f"\nUser:{question}\nBot:{full}\n"
        )[-6000:]

        _bounded_set(sessions, session, updated_history, MAX_SESSIONS)

    return StreamingResponse(generator(), media_type="text/plain")


# ==========================================
# REBUILD DATABASE
# ==========================================

@app.post("/rebuild_db")

def rebuild():

    text = open("data/all_weeks.txt", encoding="utf-8").read()

    chunks = chunk_text(text)

    emb = embed.encode(chunks, normalize_embeddings=True)

    idx = faiss.IndexFlatIP(emb.shape[1])

    idx.add(emb)

    faiss.write_index(idx, DB_INDEX)

    pickle.dump(chunks, open(DB_META, "wb"))

    load_db()

    return {
        "status": "database rebuilt",
        "chunks": len(chunks)
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")

def root():

    return {
        "status": "ELITE RAG RUNNING",
        "chunks": len(meta),
        "cache": len(cache),
        "sessions": len(sessions)
    }
