import os, pickle, faiss
import re
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.chunker import chunk_text


# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL="all-MiniLM-L6-v2"
DB_INDEX="database/faiss.index"
DB_META="database/meta.pkl"
FALLBACK="The question is not in the study material."

MAX_CACHE_ITEMS = 200
MAX_SESSIONS = 200

app = FastAPI(title="ELITE RAG API")

SYSTEM = """
You are a professional Machine Learning tutor.

STRICT RULES (MANDATORY):
1. Answer ONLY from STUDY MATERIAL provided in the prompt.
2. Do NOT use outside knowledge, assumptions, or prior model knowledge.
3. If STUDY MATERIAL is missing, weak, or does not contain the answer, reply EXACTLY:
The question is not in the study material.
4. Never mention file names, pages, retrieval, chunks, or context blocks.
5. Keep answers clear, structured, and student-friendly.
"""


# ==========================================
# GLOBAL OBJECTS
# ==========================================

embed = SentenceTransformer(MODEL)

index=None
meta=None

cache=OrderedDict()         # answer cache
cache_vectors=OrderedDict() # semantic cache vectors
sessions=OrderedDict()      # chat memory


# ==========================================
# RESET RUNTIME STATE
# ==========================================

def reset_runtime_state():
    cache.clear()
    cache_vectors.clear()
    sessions.clear()


def _bounded_set(store: OrderedDict, key, value, max_items: int):
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
    index=faiss.read_index(DB_INDEX)
    meta=pickle.load(open(DB_META,"rb"))
    # Avoid serving stale answers after DB updates/reloads.
    reset_runtime_state()

load_db()


# ==========================================
# RETRIEVE (AUTO-K)
# ==========================================

def retrieve(q,k=None):

    if k is None:
        L=len(q.split())
        if L<5:k=3
        elif L<12:k=5
        else:k=7

    vec=embed.encode([q],normalize_embeddings=True)
    _,I=index.search(vec,k)

    return [meta[i] for i in I[0]], vec


def is_grounded(answer, docs):
    if not answer:
        return False

    if answer.strip() == FALLBACK:
        return True

    context = " ".join(docs).lower()
    answer_tokens = re.findall(r"[a-zA-Z0-9]{4,}", answer.lower())

    if not answer_tokens:
        return False

    unique_tokens = set(answer_tokens)
    hits = sum(1 for t in unique_tokens if t in context)
    ratio = hits / max(1, len(unique_tokens))

    return ratio >= 0.35


# ==========================================
# STREAM LLM
# ==========================================

def stream_answer(prompt):

    stream=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==========================================
# ASK STREAMING
# ==========================================

@app.get("/ask_stream")

def ask_stream(question:str, session:str="default"):

    if not question.strip():
        raise HTTPException(400,"Empty question")

    # ------------------------------
    # SMART SEMANTIC CACHE
    # ------------------------------
    vec=embed.encode([question],normalize_embeddings=True)

    for old_q,old_v in list(cache_vectors.items()):
        score=(vec@old_v.T)[0][0]
        if score>0.92:
            # LRU behavior: recent hit moves to end.
            cache_vectors.move_to_end(old_q)
            cache.move_to_end(old_q)
            return StreamingResponse(iter([cache[old_q]]), media_type="text/plain")

    # ------------------------------
    # RETRIEVE
    # ------------------------------
    docs,vec=retrieve(question)

    if not docs:
        return StreamingResponse(iter([FALLBACK]), media_type="text/plain")

    # hallucination guard
    if sum(len(d.split()) for d in docs)<80:
        return StreamingResponse(iter([FALLBACK]), media_type="text/plain")

    context="\n\n".join(docs[:4])

    history=sessions.get(session,"")
    if session in sessions:
        sessions.move_to_end(session)

    prompt=f"""
CHAT HISTORY:
{history}

STUDY MATERIAL:
{context}

QUESTION:
{question}

Give best student-friendly explanation.
"""

    # ------------------------------
    # STREAM GENERATOR
    # ------------------------------
    def generator():

        full="".join(stream_answer(prompt))
        if not is_grounded(full, docs):
            full = FALLBACK

        # save semantic cache
        _bounded_set(cache, question, full, MAX_CACHE_ITEMS)
        _bounded_set(cache_vectors, question, vec, MAX_CACHE_ITEMS)

        # save trimmed session memory
        updated_history = (history+f"\nUser:{question}\nBot:{full}\n")[-6000:]
        _bounded_set(sessions, session, updated_history, MAX_SESSIONS)

        yield full

    return StreamingResponse(generator(),media_type="text/plain")


# ==========================================
# REBUILD DATABASE ENDPOINT
# ==========================================

@app.post("/rebuild_db")

def rebuild():

    text=open("data/all_weeks.txt",encoding="utf-8").read()
    chunks=chunk_text(text)

    emb=embed.encode(chunks,normalize_embeddings=True)

    idx=faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    faiss.write_index(idx,DB_INDEX)
    pickle.dump(chunks,open(DB_META,"wb"))

    load_db()

    return {
        "status":"database rebuilt",
        "chunks":len(chunks),
        "cache_cleared":True,
        "sessions_cleared":True
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")

def root():
    return {
        "status":"ELITE RAG RUNNING",
        "chunks": len(meta),
        "cache": len(cache),
        "sessions": len(sessions)
    }

#! uvicorn ml_api:app --reload
