"""
rag_chain.py
============
RAG pipeline: retrieval → confidence gate → citations → LLM streaming.

The ask() function now accepts pre-loaded Chroma stores passed in from
api.py's startup event, so the embedding model is never re-loaded
mid-request (critical for staying within Render's 512MB RAM limit).
"""

from __future__ import annotations

import json
from typing import Generator, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retriever import ScoredDoc, retrieve_with_scores

# ── Fallback message ───────────────────────────────────────────────────────────

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information in my knowledge base to answer "
    "your question confidently.\n\n"
    "Please **call 611** or open the **MyTelecom app** for personalised support. "
    "Our agents are available 24/7."
)

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional telecom customer care assistant.
Your job is to resolve customer issues about mobile connectivity, billing, SIM cards, and roaming.

RULES:
1. Answer ONLY from the context provided below. Do NOT use your own training knowledge.
2. For EVERY claim you make, append the citation tag shown in the context block.
   Example: "Toggle airplane mode off and on [FAQ #2 [data]]."
3. If the context does not contain enough information, say so and tell the customer to call 611.
4. Keep your answer concise, friendly, and structured. Use bullet points for multi-step troubleshooting.

CONTEXT:
{context}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_docs_with_citations(scored_docs: List[ScoredDoc]) -> str:
    """Build the context block injected into the system prompt."""
    sections = []
    for sd in scored_docs:
        header = f"[{sd.citation}]  (relevance score: {sd.score:.3f})"
        sections.append(f"{header}\n{sd.document.page_content}")
    return "\n\n---\n\n".join(sections)


# ── Streaming generator ────────────────────────────────────────────────────────

def ask(
    question: str,
    faq_store=None,
    tickets_store=None,
    guides_store=None,
) -> Generator[str, None, None]:
    """
    Stream an answer as SSE events.

    Accepts optional pre-loaded Chroma stores from api.py startup.
    Falls back to building its own stores if called from main.py CLI
    (where stores are not pre-loaded).

    SSE event types emitted:
      {"type": "sources",  "data": [{citation, score, preview}]}
      {"type": "token",    "data": "<token>"}
      {"type": "fallback", "data": "<message>"}
    """

    # ── Retrieve scored documents ──────────────────────────────────────────────
    scored_docs = retrieve_with_scores(
        question,
        faq_store=faq_store,
        tickets_store=tickets_store,
        guides_store=guides_store,
    )

    # ── Confidence gate ────────────────────────────────────────────────────────
    if not scored_docs:
        yield f"data: {json.dumps({'type': 'fallback', 'data': FALLBACK_MESSAGE})}\n\n"
        return

    # ── Emit source metadata for the UI citations panel ───────────────────────
    sources_payload = [
        {
            "citation": sd.citation,
            "score":    round(sd.score, 4),
            "preview":  sd.document.page_content[:120] + "…",
        }
        for sd in scored_docs
    ]
    yield f"data: {json.dumps({'type': 'sources', 'data': sources_payload})}\n\n"

    # ── Build prompt and stream LLM tokens ────────────────────────────────────
    context = _format_docs_with_citations(scored_docs)

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    for token in chain.stream({"context": context, "question": question}):
        yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"


# ── CLI chain (used by main.py) ────────────────────────────────────────────────

def build_chain():
    """Returns a simple callable for the CLI that prints streamed output."""
    from langchain_core.runnables import RunnableLambda

    def pipeline(question: str) -> str:
        scored_docs = retrieve_with_scores(question)
        if not scored_docs:
            return FALLBACK_MESSAGE
        context = _format_docs_with_citations(scored_docs)
        llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, reasoning_format="parsed")
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        return (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return RunnableLambda(pipeline)
