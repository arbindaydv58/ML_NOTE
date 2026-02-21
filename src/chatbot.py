import os
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

RULES:

1. First check if STUDY MATERIAL contains relevant info.
2. If yes:
   - Use study material as the main source
   - You may use your knowledge to improve explanation
   - Give a clear structured teaching-style answer.
3. If NOT:
   reply EXACTLY:
   The question is not in the study material.
4. Never mention file names, pages, or context blocks.
5. Always explain clearly for students.
"""


# ----------------------------
# CLEAN OCR NOISE
# ----------------------------
def clean_chunks(docs):
    out=[]
    for d in docs:
        d=d.replace("FILE:","")
        d=d.replace("PAGE:","")
        d=d.replace("==============================","")
        d=d.replace("====","")
        out.append(d.strip())
    return out


# ----------------------------
# SIMPLE CONFIDENCE CHECK
# ----------------------------
def context_strength(docs):

    if not docs:
        return 0

    total=0
    for d in docs[:3]:
        total+=len(d.split())

    return total


# ----------------------------
# FINAL ASK FUNCTION
# ----------------------------
def ask(question,debug=False):

    # ----------------------------
    # INPUT SAFETY
    # ----------------------------
    if not question or not question.strip():
        return "Please ask a valid question."

    if len(question) > 600:
        return "Question too long."

    # ----------------------------
    # RETRIEVE
    # ----------------------------
    docs = retrieve(question)

    if not docs:
        return "The question is not in the study material."

    docs = rerank(question,docs)

    # ----------------------------
    # CONTEXT QUALITY CHECK
    # ----------------------------
    strength=context_strength(docs)

    # If retrieved text too small → reject
    if strength < 60:
        return "The question is not in the study material."

    # ----------------------------
    # ADAPTIVE CHUNK COUNT
    # ----------------------------
    if strength > 400:
        use_k=5
    elif strength > 200:
        use_k=4
    else:
        use_k=3

    docs = clean_chunks(docs[:use_k])
    context="\n\n".join(docs)

    # ----------------------------
    # CHAT MEMORY
    # ----------------------------
    history=memory.format()

    prompt=f"""
CHAT HISTORY:
{history}

STUDY MATERIAL:
{context}

QUESTION:
{question}

Give the BEST clear student-friendly explanation.
"""

    # ----------------------------
    # LLM CALL
    # ----------------------------
    r=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt}
        ],
        temperature=0.25
    )

    answer=r.choices[0].message.content.strip()

    memory.add(question,answer)

    # ----------------------------
    # DEBUG MODE
    # ----------------------------
    if debug:

        print("\n--- RETRIEVAL STRENGTH:",strength,"---")

        print("\n--- USED CHUNKS ---")
        for i,c in enumerate(docs):
            print(f"\nChunk {i+1}:\n{c[:400]}")

    return answer