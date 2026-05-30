"""
retriever.py
============
Builds a score-aware retriever across three Chroma vector collections:

  ┌─────────────┬──────────────────────────────────────────────────────┐
  │ Collection  │ Source                                               │
  ├─────────────┼──────────────────────────────────────────────────────┤
  │ faq         │ faq.csv  — 1 document per FAQ row (no chunking)     │
  │ tickets     │ tickets.db — 1 document per resolved ticket          │
  │ guides      │ telecom_guide.pdf — 600-char chunks, 100-char overlap│
  └─────────────┴──────────────────────────────────────────────────────┘

Key upgrade over the original:
  • Uses similarity_search_with_score() instead of a plain retriever so
    every returned document carries a relevance score (L2 distance from
    Chroma; lower = more similar).
  • Applies a configurable SIMILARITY_THRESHOLD — chunks that are "too
    far" from the query are silently dropped, preventing the LLM from
    hallucinating answers from irrelevant context.
  • Attaches rich citation metadata to each Document so rag_chain.py can
    surface "FAQ #3", "Ticket TK-007", or "Guide p.4" in the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ── Constants ──────────────────────────────────────────────────────────────────

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma returns L2 (Euclidean) distances — smaller means more similar.
# Chunks with distance > threshold are considered "off-topic" and discarded.
# Tune this value based on your dataset; 1.0 is a reasonable starting point
# for all-MiniLM-L6-v2 on typical customer-care queries.
SIMILARITY_THRESHOLD = 2.0


# ── Public data contract ───────────────────────────────────────────────────────

@dataclass
class ScoredDoc:
    """A retrieved document paired with its similarity score and a human-
    readable citation string that the LLM will embed in its answer."""
    document: Document
    score: float          # L2 distance (lower = better match)
    citation: str         # e.g. "FAQ #12", "Ticket TK-007", "Guide p.3 §2"


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_faq_citation(doc: Document) -> str:
    """Build a short, human-readable citation for an FAQ entry."""
    faq_id   = doc.metadata.get("faq_id", "?")
    category = doc.metadata.get("category", "general")
    return f"FAQ #{faq_id} [{category}]"


def _make_ticket_citation(doc: Document) -> str:
    """Build a short citation for a resolved support ticket."""
    ticket_id = doc.metadata.get("ticket_id", "???")
    category  = doc.metadata.get("category", "general")
    return f"Ticket {ticket_id} [{category}]"


def _make_guide_citation(doc: Document) -> str:
    """Build a short citation for a PDF guide chunk, including page number."""
    # PyPDFLoader injects 'page' (0-indexed) into metadata
    page  = doc.metadata.get("page", None)
    chunk = doc.metadata.get("chunk_index", "?")
    if page is not None:
        return f"Guide p.{int(page) + 1} chunk {chunk}"
    return f"Guide chunk {chunk}"


def _score_and_filter(
    store: Chroma,
    query: str,
    k: int,
    citation_fn,
    threshold: float,
) -> List[ScoredDoc]:
    """
    Query a single Chroma collection with scores, filter by threshold, and
    attach citations.

    Args:
        store:       The Chroma vectorstore to query.
        query:       The user's natural-language question.
        k:           Maximum number of results to retrieve per collection.
        citation_fn: Callable that generates a citation string from a Document.
        threshold:   Maximum allowed L2 distance. Docs above this are dropped.

    Returns:
        A list of ScoredDoc objects, sorted by ascending score (best first).
    """
    # similarity_search_with_score returns List[Tuple[Document, float]]
    # where float is the L2 distance (lower = closer = more relevant).
    raw_results: List[Tuple[Document, float]] = store.similarity_search_with_score(
        query, k=k
    )

    scored = []
    for doc, score in raw_results:
        if score <= threshold:                       # keep only relevant hits
            scored.append(ScoredDoc(
                document=doc,
                score=score,
                citation=citation_fn(doc),
            ))

    # Sort ascending so the most relevant result comes first
    scored.sort(key=lambda sd: sd.score)
    return scored


# ── Public factory ─────────────────────────────────────────────────────────────

def retrieve_with_scores(
    query: str,
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[ScoredDoc]:
    """
    Retrieve the most relevant documents for *query* from all three collections.

    Returns a flat list of ScoredDoc items.  If the list is empty the caller
    (rag_chain.py) should skip the LLM and return the fallback message instead.

    Args:
        query:     The customer's question.
        k_faq:     Max docs to pull from the FAQ collection.
        k_tickets: Max docs to pull from the tickets collection.
        k_guides:  Max docs to pull from the guides (PDF) collection.
        threshold: L2 distance cut-off; documents further than this are dropped.
    """
    # Build the shared embedding model once — HuggingFaceEmbeddings caches
    # the underlying tokeniser and model weights in memory.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # ── Open each persisted Chroma collection ──────────────────────────────────
    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # ── Query each collection and merge results ────────────────────────────────
    results: List[ScoredDoc] = (
        _score_and_filter(faq_store,     query, k_faq,     _make_faq_citation,    threshold)
        + _score_and_filter(tickets_store, query, k_tickets, _make_ticket_citation, threshold)
        + _score_and_filter(guides_store,  query, k_guides,  _make_guide_citation,  threshold)
    )

    return results
