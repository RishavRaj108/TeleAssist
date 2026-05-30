"""
rag_chain.py
============
Assembles the full RAG pipeline:

    User question
         │
         ▼
    retrieve_with_scores()          ← retriever.py
         │
    ┌────┴──────────────────────────────────────────┐
    │ Confidence gate                                │
    │  • All scores > threshold  →  fallback reply  │
    │  • At least one good hit   →  continue        │
    └────┬──────────────────────────────────────────┘
         │
    _format_docs_with_citations()   ← builds context block with inline refs
         │
         ▼
    ChatPromptTemplate → Qwen3-32B (Groq) → StrOutputParser
         │
         ▼
    Streamed answer (with citation markers)

Two public entry points
-----------------------
  build_chain()  – returns a LangChain Runnable for the CLI (main.py)
  ask(question)  – called by the FastAPI server; returns a generator for SSE streaming
"""

from __future__ import annotations

import json
from typing import Generator, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from retriever import ScoredDoc, retrieve_with_scores

# ── Fallback message ───────────────────────────────────────────────────────────

# Returned verbatim when no retrieved chunk clears the confidence threshold.
# This avoids the LLM speculating from zero context.
FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information in my knowledge base to answer "
    "your question confidently.\n\n"
    "Please **call 611** or open the **MyTelecom app** for personalised support. "
    "Our agents are available 24/7."
)

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional telecom customer care assistant.
Your job is to resolve customer issues about mobile connectivity, billing, SIM cards, and roaming.

════════════════════════════════════════════════
RULES — read carefully:
════════════════════════════════════════════════

1. Answer ONLY from the context provided below.
   Do NOT use your own training knowledge to fill gaps.

2. For EVERY claim you make, append the citation tag shown in the context block.
   Example:
     "Toggle airplane mode off and on to force a reconnect [FAQ #2 [data]]."
     "Your SIM was replaced successfully in a similar case [Ticket TK-012 [sim]]."
     "APN settings control how your device reaches the carrier's data network [Guide p.3 chunk 4]."

3. If the context does not contain enough information to answer fully,
   say so clearly and tell the customer to call 611 or use the MyTelecom app.

4. Keep your answer concise, friendly, and structured.
   Use bullet points for multi-step troubleshooting.

════════════════════════════════════════════════
CONTEXT (retrieved knowledge):
════════════════════════════════════════════════

{context}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_docs_with_citations(scored_docs: List[ScoredDoc]) -> str:
    """
    Convert a list of ScoredDoc objects into the structured context block that
    gets injected into the system prompt.

    Each block header shows the citation string and the similarity score so the
    LLM understands which sources are most relevant.  The LLM is instructed to
    echo these citation strings inside its answer.

    Example output
    --------------
    [FAQ #2 [data]]  (score: 0.412)
    Q: Why is my mobile internet so slow?
    A: Slow speeds are usually caused by …

    ---

    [Ticket TK-007 [connectivity]]  (score: 0.531)
    Issue: No internet access
    …
    """
    sections = []
    for sd in scored_docs:
        header = f"[{sd.citation}]  (relevance score: {sd.score:.3f})"
        sections.append(f"{header}\n{sd.document.page_content}")
    return "\n\n---\n\n".join(sections)


def _has_confident_results(scored_docs: List[ScoredDoc]) -> bool:
    """Return True if at least one retrieved document clears the threshold.

    retrieve_with_scores() already filters by threshold, so an empty list
    means nothing was confident enough.
    """
    return len(scored_docs) > 0


# ── LangChain Runnable (used by CLI / main.py) ─────────────────────────────────

def build_chain():
    """
    Build and return a LangChain Runnable chain for CLI use.

    The chain signature:  str  →  str  (supports .stream())

    The chain handles the confidence gate internally via a RunnableLambda
    that either short-circuits to the fallback or runs the full LLM pipeline.
    """

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,          # deterministic — important for support contexts
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    output_parser = StrOutputParser()

    def pipeline(question: str) -> str:
        """Inner function that wires retrieval → gate → LLM."""
        scored_docs = retrieve_with_scores(question)

        # ── Confidence gate ────────────────────────────────────────────────────
        if not _has_confident_results(scored_docs):
            return FALLBACK_MESSAGE

        # ── Build context and run LLM ──────────────────────────────────────────
        context = _format_docs_with_citations(scored_docs)
        chain   = prompt | llm | output_parser
        return chain.invoke({"context": context, "question": question})

    return RunnableLambda(pipeline)


# ── Streaming generator (used by FastAPI / api.py) ────────────────────────────

def ask(question: str) -> Generator[str, None, None]:
    """
    Stream an answer token-by-token as Server-Sent Events (SSE).

    The FastAPI endpoint iterates over this generator and forwards each
    yielded string to the browser.  Three special event types are emitted:

      {"type": "sources", "data": [...]}   — sent BEFORE the first token;
                                             contains citation metadata for
                                             the frontend to render a sources panel.

      {"type": "token",   "data": "..."}   — one streamed token from the LLM.

      {"type": "fallback","data": "..."}   — emitted instead of tokens when the
                                             confidence gate fires.

    Yields:
        JSON-serialised SSE strings, each terminated with a double newline
        so the browser's EventSource API receives clean events.
    """
    # ── Step 1: Retrieve scored documents ─────────────────────────────────────
    scored_docs = retrieve_with_scores(question)

    # ── Step 2: Confidence gate ────────────────────────────────────────────────
    if not _has_confident_results(scored_docs):
        payload = json.dumps({"type": "fallback", "data": FALLBACK_MESSAGE})
        yield f"data: {payload}\n\n"
        return

    # ── Step 3: Emit source metadata so the UI can render a citations panel ───
    sources_payload = [
        {
            "citation": sd.citation,
            "score":    round(sd.score, 4),
            "preview":  sd.document.page_content[:120] + "…",  # short teaser
        }
        for sd in scored_docs
    ]
    yield f"data: {json.dumps({'type': 'sources', 'data': sources_payload})}\n\n"

    # ── Step 4: Build prompt context and stream LLM tokens ────────────────────
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
        payload = json.dumps({"type": "token", "data": token})
        yield f"data: {payload}\n\n"
