# ML NOTE CHATBOT

RAG chatbot for Machine Learning notes using FAISS + sentence-transformers + Groq LLM.

## What Is Implemented

- Note-only retrieval pipeline from your provided study material.
- Strict guardrails in both CLI and API:
  - No outside knowledge allowed.
  - If answer is not supported by notes, response is exactly:
    `The question is not in the study material.`
- Retrieval + reranking:
  - Dense retrieval with `all-MiniLM-L6-v2`
  - Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Confidence score display.
- Streaming API responses.
- Semantic cache + session memory in API.
- Automatic cache/session reset on DB reload and `/rebuild_db`.

## Project Structure

- `src/build_db.py`: Build FAISS index from `data/all_weeks.txt`
- `src/chatbot.py`: Core note-only chatbot logic
- `app.py`: Terminal chat app
- `ml_api.py`: FastAPI streaming API
- `database/faiss.index` and `database/meta.pkl`: Vector DB files

## Prerequisites

- Python 3.10+
- Groq API key

Create `.env` in project root:

```env
GROQ_API_KEY=your_key_here
```

## Install Dependencies

```powershell
pip install python-dotenv groq sentence-transformers faiss-cpu fastapi uvicorn
```

## Build / Rebuild Vector Database

```powershell
python -m src.build_db
```

This reads `data/all_weeks.txt`, chunks content, embeds chunks, and writes:
- `database/faiss.index`
- `database/meta.pkl`

## Start CLI Chatbot

```powershell
python app.py
```

CLI commands:
- `debug on`
- `debug off`
- `exit`

## Start API Server

```powershell
uvicorn ml_api:app --reload
```

API endpoints:
- `GET /` health and runtime stats
- `GET /ask_stream?question=...&session=default` streaming answer
- `POST /rebuild_db` rebuild index from notes and clear cache/session

## Notes

- If you change study material, run `python -m src.build_db` or call `POST /rebuild_db`.
- Strict mode is active in both CLI and API (`temperature=0.0`, note-only prompt rules).
