"""
api.py
======
FastAPI server with startup pre-loading of embeddings and Chroma stores.

KEY FIX: On Render's free tier (512MB RAM), loading the embedding model
on every request causes OOM crashes. We load it ONCE at startup and reuse
it for all requests. This cuts per-request RAM from ~400MB to ~50MB.
"""

from __future__ import annotations
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Telecom RAG Chatbot API", version="2.0.0")

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://tele-assist-one.vercel.app",  # no trailing slash
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]

# ── Global stores: loaded ONCE at startup, reused for every request ────────────
# This is the critical fix — keeping these in module-level globals means the
# 90MB embedding model is loaded exactly once when uvicorn starts, not on
# every incoming request. Cuts per-request RAM from ~400MB to ~50MB.

_embeddings    = None
_faq_store     = None
_tickets_store = None
_guides_store  = None


@app.on_event("startup")
def load_models():
    """Load embedding model and open Chroma collections once at server startup."""
    global _embeddings, _faq_store, _tickets_store, _guides_store

    print("[startup] Loading embedding model...")
    from langchain_huggingface import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[startup] Embedding model ready.")

    from langchain_chroma import Chroma
    CHROMA_DIR = "chroma_store"

    print("[startup] Opening Chroma collections...")
    _faq_store     = Chroma(collection_name="faq",     embedding_function=_embeddings, persist_directory=CHROMA_DIR)
    _tickets_store = Chroma(collection_name="tickets", embedding_function=_embeddings, persist_directory=CHROMA_DIR)
    _guides_store  = Chroma(collection_name="guides",  embedding_function=_embeddings, persist_directory=CHROMA_DIR)

    print(
        f"[startup] Ready — "
        f"faq={_faq_store._collection.count()} "
        f"tickets={_tickets_store._collection.count()} "
        f"guides={_guides_store._collection.count()} vectors"
    )


# ── Pydantic ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample-questions")
def sample_questions():
    return {"questions": SAMPLE_QUESTIONS}


@app.post("/chat")
def chat(request: ChatRequest):
    """SSE streaming chat endpoint — passes pre-loaded stores to rag_chain."""

    def event_stream():
        try:
            from rag_chain import ask
            for event in ask(
                request.question,
                faq_store=_faq_store,
                tickets_store=_tickets_store,
                guides_store=_guides_store,
            ):
                yield event
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
