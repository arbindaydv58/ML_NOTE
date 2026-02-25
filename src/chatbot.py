import os
import time
from dotenv import load_dotenv
from groq import Groq
from src.retriever import retrieve
from src.reranker import rerank
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


# -----------------------------
# CONFIDENCE CALCULATION
# -----------------------------
def calc_confidence(docs):

    if not docs:
        return 0

    words=sum(len(d.split()) for d in docs[:3])

    if words>800: return 95
    if words>500: return 88
    if words>300: return 80
    if words>150: return 70
    if words>80:  return 60
    return 40


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

    memory.add(question,answer)

    confidence=calc_confidence(docs)

    # ---------- DEBUG ----------
    if debug:

        print("\n--- RETRIEVAL WORDS:",strength,"---")

        for i,c in enumerate(docs):
            print(f"\nChunk {i+1}:\n{c[:350]}")

    # ---------- STREAM OUTPUT ----------
    stream(answer)

    print(f"\nConfidence: {confidence}%")

    return ""
