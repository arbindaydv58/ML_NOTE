# ML NOTE CHATBOT

RAG chatbot for Machine Learning notes using FAISS + sentence-transformers + Groq LLM.

## LinkedIn Share Note

Built **ML NOTE CHATBOT**: a note-grounded RAG assistant for Machine Learning study material.

- Retrieval: FAISS + `all-MiniLM-L6-v2`
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Interface: CLI + FastAPI + Vite React (TypeScript)
- Guardrail: if answer is unsupported, it returns exactly:
  `The question is not in the study material.`

Focused on strict grounding, clean retrieval, and student-friendly explanations.

## Current Features

- Note-only retrieval pipeline from your study material (`data/all_weeks.txt`).
- Strict guardrails in CLI and API:
  - No outside knowledge allowed.
  - Unsupported answers are forced to:
    `The question is not in the study material.`
- Grounding check after generation (extra hallucination control).
- Retrieval + reranking:
  - Dense retrieval with `all-MiniLM-L6-v2`
  - Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- FastAPI with:
  - Session memory
  - Semantic cache
  - Bounded cache/session storage (LRU-style behavior)
  - `/rebuild_db` support
- CLI warmup: models load at app startup, not after first question.
- Vite React TypeScript frontend with ChatGPT-style typewriter answer rendering.

## Project Structure

- `app.py`: Terminal chat app
- `ml_api.py`: FastAPI app
- `src/chatbot.py`: Core chatbot logic + grounding check
- `src/retriever.py`: FAISS retrieval (lazy load + warmup helper)
- `src/reranker.py`: Cross-encoder reranking (lazy load + warmup helper)
- `src/chunker.py`: Chunk generation with overlap support
- `src/build_db.py`: Build FAISS index from notes
- `clean_dataset.py`: Clean extracted note text
- `convert_all_pdfs.py`: Extract/OCR PDFs into combined text
- `database/faiss.index`, `database/meta.pkl`: Vector DB files
- `frontend/`: Vite React TypeScript UI

## Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API key

Create `.env` in project root:

```env
GROQ_API_KEY=your_key_here
```

## Install Backend Dependencies

```powershell
pip install python-dotenv groq sentence-transformers faiss-cpu fastapi uvicorn python-multipart
```

## Build / Rebuild Vector Database

```powershell
python -m src.build_db
```

This reads `data/all_weeks.txt` and writes:
- `database/faiss.index`
- `database/meta.pkl`

## Run CLI Chatbot

```powershell
python app.py
```

CLI commands:
- `debug on`
- `debug off`
- `exit`

## Run FastAPI Backend

```powershell
uvicorn ml_api:app --reload
```

API endpoints:
- `GET /` health and runtime stats
- `GET /ask_stream?question=...&session=default`
- `POST /rebuild_db`

## Run Frontend (Vite React + TypeScript)

```powershell
cd frontend
npm install
npm run dev
```

Open:
- `http://localhost:5173`

Frontend calls backend at:
- `http://127.0.0.1:8000`

## Notes

- If study material changes, run `python -m src.build_db` or call `POST /rebuild_db`.
- First model download may take time; later runs are faster.
- HF Hub warnings about unauthenticated requests are normal without `HF_TOKEN`.
