import os
import re
import time
from dotenv import load_dotenv
from groq import Groq
from src.retriever import retrieve, warmup_retriever
from src.reranker import rerank, warmup_reranker
from src.memory import Memory

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
memory = Memory()


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


# -----------------------------
# CLEAN OCR GARBAGE
# -----------------------------
def clean_chunks(docs):

    cleaned=[]

    for d in docs:

        d=d.replace("FILE:","")
        d=d.replace("PAGE:","")
        d=d.replace("==============================","")
        d=d.replace("====","")

        cleaned.append(d.strip())

    return cleaned


FALLBACK = "The question is not in the study material."


def warmup_models():
    warmup_retriever()
    warmup_reranker()


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


# -----------------------------
# STREAM OUTPUT 
# -----------------------------
def stream(text):

    for ch in text:
        print(ch,end="",flush=True)
        time.sleep(0.01)

    print()


# -----------------------------
# MAIN ASK FUNCTION
# -----------------------------
def ask(question,debug=False):

    # ---------- INPUT VALIDATION ----------
    if not question or not question.strip():
        return "Please ask a valid question."

    if len(question)>600:
        return "Question too long."

    # ---------- RETRIEVE ----------
    docs = retrieve(question)

    if not docs:
        return "The question is not in the study material."

    docs = rerank(question,docs)

    # ---------- CONTEXT QUALITY ----------
    strength=sum(len(d.split()) for d in docs[:3])

    if strength<60:
        return "The question is not in the study material."

    # ---------- ADAPTIVE CONTEXT SIZE ----------
    if strength>500:
        use_k=5
    elif strength>250:
        use_k=4
    else:
        use_k=3

    docs=clean_chunks(docs[:use_k])

    context="\n\n".join(docs)

    # ---------- MEMORY ----------
    history=memory.format()

    prompt=f"""
CHAT HISTORY:
{history}

STUDY MATERIAL:
{context}

QUESTION:
{question}

Give the BEST clear structured student-friendly explanation.
"""

    # ---------- LLM ----------
    r=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt}
        ],
        temperature=0.0
    )

    answer=r.choices[0].message.content.strip()

    if not is_grounded(answer, docs):
        answer = FALLBACK

    memory.add(question,answer)

    # ---------- DEBUG ----------
    if debug:

        print("\n--- RETRIEVAL WORDS:",strength,"---")

        for i,c in enumerate(docs):
            print(f"\nChunk {i+1}:\n{c[:350]}")

    # ---------- STREAM OUTPUT ----------
    stream(answer)

    return ""
