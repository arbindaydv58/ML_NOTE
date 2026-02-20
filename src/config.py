import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH","data")
EMBED_MODEL = os.getenv("EMBED_MODEL","all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")

TOP_K = int(os.getenv("TOP_K",20))
FINAL_K = int(os.getenv("FINAL_K",5))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")