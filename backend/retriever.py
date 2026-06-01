"""
retriever.py
============
Score-aware retrieval across three Chroma collections.

Updated to accept pre-loaded stores passed in from api.py (startup pre-loading)
so the embedding model is never initialised more than once per server process.
Falls back to building its own stores when called from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 1.0   # L2 distance cut-off; raise if getting too many fallbacks


@dataclass
class ScoredDoc:
    """A retrieved document with its similarity score and citation string."""
    document: Document
    score: float
    citation: str


# ── Citation builders ──────────────────────────────────────────────────────────

def _make_faq_citation(doc: Document) -> str:
    return f"FAQ #{doc.metadata.get('faq_id', '?')} [{doc.metadata.get('category', 'general')}]"

def _make_ticket_citation(doc: Document) -> str:
    return f"Ticket {doc.metadata.get('ticket_id', '???')} [{doc.metadata.get('category', 'general')}]"

def _make_guide_citation(doc: Document) -> str:
    page  = doc.metadata.get("page", None)
    chunk = doc.metadata.get("chunk_index", "?")
    return f"Guide p.{int(page) + 1} chunk {chunk}" if page is not None else f"Guide chunk {chunk}"


# ── Core retrieval ─────────────────────────────────────────────────────────────

def _score_and_filter(store, query: str, k: int, citation_fn, threshold: float) -> List[ScoredDoc]:
    """Query one Chroma store with scores, filter by threshold, attach citations."""
    raw: List[Tuple[Document, float]] = store.similarity_search_with_score(query, k=k)
    results = [
        ScoredDoc(document=doc, score=score, citation=citation_fn(doc))
        for doc, score in raw
        if score <= threshold
    ]
    results.sort(key=lambda sd: sd.score)
    return results


def retrieve_with_scores(
    query: str,
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
    threshold: float = SIMILARITY_THRESHOLD,
    faq_store=None,
    tickets_store=None,
    guides_store=None,
) -> List[ScoredDoc]:
    """
    Retrieve relevant documents from all three collections.

    Uses pre-loaded stores if provided (from api.py startup).
    Builds its own stores if called from the CLI (main.py).
    """
    # Build stores only if not pre-loaded (CLI mode)
    if faq_store is None or tickets_store is None or guides_store is None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        )
        faq_store     = Chroma(collection_name="faq",     embedding_function=embeddings, persist_directory=CHROMA_DIR)
        tickets_store = Chroma(collection_name="tickets", embedding_function=embeddings, persist_directory=CHROMA_DIR)
        guides_store  = Chroma(collection_name="guides",  embedding_function=embeddings, persist_directory=CHROMA_DIR)

    return (
        _score_and_filter(faq_store,     query, k_faq,     _make_faq_citation,    threshold)
        + _score_and_filter(tickets_store, query, k_tickets, _make_ticket_citation, threshold)
        + _score_and_filter(guides_store,  query, k_guides,  _make_guide_citation,  threshold)
    )
