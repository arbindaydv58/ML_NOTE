# ==========================================
# ELITE RAG API (INDUSTRY STYLE)
# Streaming + Semantic Cache + Memory + DB rebuild
# ==========================================

import os, pickle, faiss
from typing import Dict
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

cache={}                 # answer cache
cache_vectors={}         # semantic cache vectors
sessions={}              # chat memory


# ==========================================
# RESET RUNTIME STATE
# ==========================================

def reset_runtime_state():
    cache.clear()
    cache_vectors.clear()
    sessions.clear()


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
    D,I=index.search(vec,k)

    return [meta[i] for i in I[0]], vec


# ==========================================
# CONFIDENCE (BETTER)
# ==========================================

def confidence(docs):

    if not docs:
        return 0

    base=sum(len(d.split()) for d in docs[:3])

    if base>600:return 95
    if base>400:return 88
    if base>250:return 78
    if base>120:return 65
    return 50


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

    for old_q,old_v in cache_vectors.items():
        score=(vec@old_v.T)[0][0]
        if score>0.92:
            return StreamingResponse(iter([cache[old_q]]))

    # ------------------------------
    # RETRIEVE
    # ------------------------------
    docs,vec=retrieve(question)

    if not docs:
        return StreamingResponse(iter(["The question is not in the study material."]))

    # hallucination guard
    if sum(len(d.split()) for d in docs)<80:
        return StreamingResponse(iter(["The question is not in the study material."]))

    context="\n\n".join(docs[:4])

    history=sessions.get(session,"")

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

        full=""

        # heartbeat prevents frontend freeze
        yield ""

        for token in stream_answer(prompt):
            full+=token
            yield token

        # save semantic cache
        cache[question]=full
        cache_vectors[question]=vec

        # save trimmed session memory
        sessions[session]=(history+f"\nUser:{question}\nBot:{full}\n")[-6000:]

        conf=confidence(docs)
        yield f"\n\n[Confidence: {conf}%]"

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
