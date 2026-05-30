"""
api.py
======
FastAPI server — the bridge between the React frontend and the RAG backend.

Endpoints
---------
  GET  /health           — liveness probe (returns {"status": "ok"})
  POST /chat             — SSE streaming endpoint; receives a question,
                           streams citation metadata + LLM tokens back to the
                           browser as Server-Sent Events.
  GET  /sample-questions — returns the list of pre-canned sample questions
                           so the sidebar is driven by the server, not hardcoded
                           in the React app.

Server-Sent Events (SSE) format
---------------------------------
Each event is a JSON object with a "type" field:

  {"type": "sources",  "data": [{citation, score, preview}, …]}
  {"type": "token",    "data": "<one streamed token>"}
  {"type": "fallback", "data": "<canned 'I don't know' message>"}
  {"type": "error",    "data": "<error message>"}
  {"type": "done"}      ← marks end of stream

Usage
-----
  uvicorn api:app --reload --port 8000

CORS is configured for localhost:5173 (Vite dev server) and localhost:3000.
Adjust ALLOWED_ORIGINS before deploying to production.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env so GROQ_API_KEY and HF_TOKEN are available before any LangChain
# code tries to read them.
load_dotenv()

# Suppress noisy HuggingFace/Transformers logs in the terminal.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rag_chain import ask  # noqa: E402 — import after env vars are set

# ── Application setup ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom RAG Chatbot API",
    description="RAG-powered telecom customer care — FastAPI + Qwen3-32B on Groq",
    version="2.0.0",
)

# Allow the React dev server (Vite) and any standard localhost port to call this API.
ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite default
    "http://localhost:3000",   # CRA / other
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Sample questions (sidebar content driven from server) ──────────────────────

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

# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    """Simple liveness probe used by Docker health checks and monitoring."""
    return {"status": "ok"}


@app.get("/sample-questions")
def sample_questions() -> dict:
    """Return the list of sample questions for the sidebar."""
    return {"questions": SAMPLE_QUESTIONS}


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """
    Main chat endpoint.  Accepts a customer question and returns a streaming
    response using Server-Sent Events (text/event-stream).

    The generator from rag_chain.ask() is forwarded directly to the browser.
    A final {"type": "done"} event is appended so the client knows streaming
    has finished and can re-enable the input field.

    Error handling: any exception inside the generator is caught here and
    serialised as a {"type": "error"} event so the UI can display it gracefully
    rather than leaving the stream hanging.
    """

    def event_stream():
        try:
            for event in ask(request.question):
                yield event
        except Exception as exc:
            error_payload = json.dumps({"type": "error", "data": str(exc)})
            yield f"data: {error_payload}\n\n"
        finally:
            # Always close the stream cleanly so the browser doesn't wait forever.
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent proxies / Nginx from buffering the SSE stream.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
