"""
rag.py — RAG (Retrieval-Augmented Generation) pipeline for EthicalGuard.

What is RAG?
------------
RAG combines two ideas:
  1. Retrieval  — find the most relevant passages from a document collection
                  using semantic (meaning-based) search.
  2. Generation — feed those passages as context to a language model so it
                  can answer questions grounded in the actual document.

This is better than asking the LM directly because:
  - The LM doesn't hallucinate facts it never saw.
  - Answers are traceable back to specific document chunks.
  - Works with any document without retraining.

Architecture
------------
                 ┌──────────────┐
  Document ───► │  ingestion   │ ──► chunks
                └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  SBERT embed │ ──► 384-dim vectors
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    FAISS     │ ──► persisted vector index (vectordb/)
                 └──────────────┘
                        ▲
  Question ──► embed ───┘ retrieve top-k chunks
                        │
                        ▼
                 ┌──────────────┐
                 │  LLM + rerank│ ──► ethical answer
                 └──────────────┘

Vector Store: FAISS
-------------------
FAISS (Facebook AI Similarity Search) is a highly optimised library for
nearest-neighbour search over dense vectors.  We use the flat L2 index
(IndexFlatL2) which performs exact search — no approximation errors.

The index and the chunk metadata (text, document name, chunk index) are
persisted to disk in the vectordb/ folder so embeddings are not recomputed
on every restart.

Embeddings: all-MiniLM-L6-v2
-----------------------------
A 384-dimensional sentence embedding model.  Fast on CPU, good semantic
quality for retrieval tasks.  Already used by scoring.py for coherence —
we reuse the same model instance to avoid loading it twice.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Optional

import faiss
import numpy as np

from app.config import VECTORDB_PATH, RAG_TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File paths for persistence
# ---------------------------------------------------------------------------
# We store three files in vectordb/:
#   index.faiss  — the FAISS index (binary, fast to load)
#   chunks.pkl   — list of raw chunk texts (parallel to index rows)
#   meta.pkl     — list of metadata dicts (document name, chunk_index)

_INDEX_FILE = os.path.join(VECTORDB_PATH, "index.faiss")
_CHUNKS_FILE = os.path.join(VECTORDB_PATH, "chunks.pkl")
_META_FILE = os.path.join(VECTORDB_PATH, "meta.pkl")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_sbert_model = None          # injected from scoring.py at startup
_faiss_index = None          # FAISS IndexFlatL2
_chunks: list[str] = []      # raw text for each vector row
_meta: list[dict] = []       # metadata for each vector row
_embedding_dim: int = 384    # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def set_rag_sbert_model(sbert_model):
    """
    Inject the already-loaded SBERT model from scoring.py.
    Called once at startup so RAG and scoring share the same model instance.
    """
    global _sbert_model
    _sbert_model = sbert_model
    logger.info("RAG: SBERT model injected.")


def init_vector_db():
    """
    Initialise the FAISS vector store.

    If persisted files exist in vectordb/, load them so previously uploaded
    documents are immediately available without re-ingestion.
    If not, create a fresh empty index.
    """
    global _faiss_index, _chunks, _meta

    os.makedirs(VECTORDB_PATH, exist_ok=True)

    if (
        os.path.exists(_INDEX_FILE)
        and os.path.exists(_CHUNKS_FILE)
        and os.path.exists(_META_FILE)
    ):
        # Load existing index from disk
        _faiss_index = faiss.read_index(_INDEX_FILE)
        with open(_CHUNKS_FILE, "rb") as f:
            _chunks = pickle.load(f)
        with open(_META_FILE, "rb") as f:
            _meta = pickle.load(f)
        logger.info(
            f"RAG: Loaded existing FAISS index with {_faiss_index.ntotal} vectors "
            f"from '{VECTORDB_PATH}'."
        )
    else:
        # Create a new flat L2 index
        # IndexFlatL2 performs exact nearest-neighbour search using Euclidean distance.
        # It's the simplest and most accurate FAISS index — fine for CPU + small corpora.
        _faiss_index = faiss.IndexFlatL2(_embedding_dim)
        _chunks = []
        _meta = []
        logger.info("RAG: Created new empty FAISS index.")


def _save_index():
    """Persist the FAISS index and metadata to disk."""
    faiss.write_index(_faiss_index, _INDEX_FILE)
    with open(_CHUNKS_FILE, "wb") as f:
        pickle.dump(_chunks, f)
    with open(_META_FILE, "wb") as f:
        pickle.dump(_meta, f)


# ---------------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------------

def add_chunks_to_db(document_name: str, chunks: list[str]) -> int:
    """
    Embed a list of text chunks and add them to the FAISS index.

    If the document was previously uploaded, its old chunks are removed first
    (upsert behaviour — no duplicates).

    Each chunk is stored as:
      - A 384-dim SBERT embedding in the FAISS index
      - The raw text in _chunks (parallel list)
      - Metadata dict in _meta (document name + chunk index)

    Returns the number of chunks added.
    """
    if _faiss_index is None:
        raise RuntimeError("Vector DB not initialised. Call init_vector_db() first.")
    if _sbert_model is None:
        raise RuntimeError("SBERT model not set. Call set_rag_sbert_model() first.")
    if not chunks:
        return 0

    # Remove existing chunks for this document (upsert)
    _remove_document_chunks(document_name)

    # Embed all chunks in one batch
    embeddings = _sbert_model.encode(chunks, convert_to_numpy=True)
    # FAISS requires float32
    embeddings = embeddings.astype(np.float32)

    # Add to FAISS index
    _faiss_index.add(embeddings)

    # Append to parallel metadata lists
    for i, chunk in enumerate(chunks):
        _chunks.append(chunk)
        _meta.append({"document": document_name, "chunk_index": i})

    # Persist to disk
    _save_index()

    logger.info(f"RAG: Added {len(chunks)} chunks for document '{document_name}'.")
    return len(chunks)


def _remove_document_chunks(document_name: str):
    """
    Remove all chunks belonging to document_name from the in-memory lists.
    Then rebuild the FAISS index from the remaining chunks.

    FAISS IndexFlatL2 does not support deletion, so we rebuild from scratch.
    This is acceptable for small-to-medium corpora (< 100k chunks).
    """
    global _faiss_index, _chunks, _meta

    # Find indices to keep
    keep_indices = [i for i, m in enumerate(_meta) if m["document"] != document_name]

    if len(keep_indices) == len(_meta):
        return  # document not present — nothing to remove

    kept_chunks = [_chunks[i] for i in keep_indices]
    kept_meta = [_meta[i] for i in keep_indices]

    # Rebuild FAISS index from kept chunks
    _faiss_index = faiss.IndexFlatL2(_embedding_dim)
    _chunks = kept_chunks
    _meta = kept_meta

    if kept_chunks:
        embeddings = _sbert_model.encode(kept_chunks, convert_to_numpy=True).astype(np.float32)
        _faiss_index.add(embeddings)

    logger.info(f"RAG: Removed chunks for document '{document_name}'.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(
    question: str,
    top_k: int = RAG_TOP_K,
    document_name: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant chunks for a question.

    How it works:
      1. Embed the question using SBERT (same model used for storage).
      2. FAISS computes L2 distance between the question embedding and every
         stored chunk embedding.
      3. Returns the top-k closest chunks.

    Parameters
    ----------
    question      : the user's question
    top_k         : number of chunks to return
    document_name : if provided, restrict results to this document

    Returns
    -------
    List of dicts with keys: text, document, chunk_index, distance
    """
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    if _sbert_model is None:
        raise RuntimeError("SBERT model not set.")

    # Embed the question
    q_emb = _sbert_model.encode([question], convert_to_numpy=True).astype(np.float32)

    # Search — retrieve more than top_k if we need to filter by document
    search_k = min(_faiss_index.ntotal, top_k * 5 if document_name else top_k)
    distances, indices = _faiss_index.search(q_emb, search_k)

    results: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        m = _meta[idx]
        # Apply document filter if requested
        if document_name and m["document"] != document_name:
            continue
        results.append({
            "text": _chunks[idx],
            "document": m["document"],
            "chunk_index": m["chunk_index"],
            "distance": round(float(dist), 4),
        })
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_documents() -> list[str]:
    """Return a sorted list of unique document names in the vector store."""
    return sorted({m["document"] for m in _meta})


def delete_document(document_name: str) -> int:
    """Remove all chunks for a document. Returns number of chunks removed."""
    before = len(_chunks)
    _remove_document_chunks(document_name)
    removed = before - len(_chunks)
    if removed:
        _save_index()
    return removed


def get_db_stats() -> dict:
    """Return basic stats about the vector store for the /rag-status endpoint."""
    if _faiss_index is None:
        return {"status": "not_initialised", "total_chunks": 0, "documents": []}
    return {
        "status": "ready",
        "total_chunks": _faiss_index.ntotal,
        "documents": list_documents(),
    }
