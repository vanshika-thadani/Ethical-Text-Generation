"""
rag.py — RAG pipeline for EthicalGuard.

Embeddings are now generated via HF Inference API (all-MiniLM-L6-v2).
No local SBERT model — no torch dependency on the server.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import faiss
import numpy as np

from app.config import VECTORDB_PATH, RAG_TOP_K

logger = logging.getLogger(__name__)

_INDEX_FILE  = os.path.join(VECTORDB_PATH, "index.faiss")
_CHUNKS_FILE = os.path.join(VECTORDB_PATH, "chunks.pkl")
_META_FILE   = os.path.join(VECTORDB_PATH, "meta.pkl")

_faiss_index = None
_chunks: list[str] = []
_meta:   list[dict] = []
_embedding_dim: int = 384


# ---------------------------------------------------------------------------
# Embedding helper (delegates to scoring._hf_embed)
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using the HF Inference API via scoring._hf_embed.
    Returns float32 numpy array of shape (len(texts), 384).
    """
    from app.scoring import _hf_embed
    vecs = _hf_embed(texts)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# No-op kept for backwards compatibility — no local SBERT to inject
# ---------------------------------------------------------------------------

def set_rag_sbert_model(_model):
    """No-op. SBERT is now called via HF Inference API in scoring._hf_embed."""
    logger.info("RAG: embeddings via HF Inference API (no local SBERT).")


# ---------------------------------------------------------------------------
# Vector DB init
# ---------------------------------------------------------------------------

def init_vector_db():
    """Load or create the FAISS flat L2 index."""
    global _faiss_index, _chunks, _meta
    os.makedirs(VECTORDB_PATH, exist_ok=True)

    if (os.path.exists(_INDEX_FILE)
            and os.path.exists(_CHUNKS_FILE)
            and os.path.exists(_META_FILE)):
        _faiss_index = faiss.read_index(_INDEX_FILE)
        with open(_CHUNKS_FILE, "rb") as f:
            _chunks = pickle.load(f)
        with open(_META_FILE, "rb") as f:
            _meta = pickle.load(f)
        logger.info(f"RAG: Loaded FAISS index with {_faiss_index.ntotal} vectors.")
    else:
        _faiss_index = faiss.IndexFlatL2(_embedding_dim)
        _chunks = []
        _meta   = []
        logger.info("RAG: Created new empty FAISS index.")


def _save_index():
    faiss.write_index(_faiss_index, _INDEX_FILE)
    with open(_CHUNKS_FILE, "wb") as f:
        pickle.dump(_chunks, f)
    with open(_META_FILE, "wb") as f:
        pickle.dump(_meta, f)


# ---------------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------------

def add_chunks_to_db(document_name: str, chunks: list[str]) -> int:
    """Embed chunks via HF API and store in FAISS."""
    if _faiss_index is None:
        raise RuntimeError("Vector DB not initialised.")
    if not chunks:
        return 0

    _remove_document_chunks(document_name)

    embeddings = _embed(chunks)
    _faiss_index.add(embeddings)

    for i, chunk in enumerate(chunks):
        _chunks.append(chunk)
        _meta.append({"document": document_name, "chunk_index": i})

    _save_index()
    logger.info(f"RAG: Added {len(chunks)} chunks for '{document_name}'.")
    return len(chunks)


def _remove_document_chunks(document_name: str):
    global _faiss_index, _chunks, _meta
    keep = [i for i, m in enumerate(_meta) if m["document"] != document_name]
    if len(keep) == len(_meta):
        return
    kept_chunks = [_chunks[i] for i in keep]
    kept_meta   = [_meta[i]   for i in keep]
    _faiss_index = faiss.IndexFlatL2(_embedding_dim)
    _chunks = kept_chunks
    _meta   = kept_meta
    if kept_chunks:
        _faiss_index.add(_embed(kept_chunks))
    logger.info(f"RAG: Removed chunks for '{document_name}'.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(
    question: str,
    top_k: int = RAG_TOP_K,
    document_name: Optional[str] = None,
) -> list[dict]:
    """Embed question via HF API and search FAISS."""
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []

    q_emb    = _embed([question])
    search_k = min(_faiss_index.ntotal, top_k * 5 if document_name else top_k)
    distances, indices = _faiss_index.search(q_emb, search_k)

    results: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_chunks):
            continue
        m = _meta[idx]
        if document_name and m["document"] != document_name:
            continue
        results.append({
            "text":        _chunks[idx],
            "document":    m["document"],
            "chunk_index": m["chunk_index"],
            "distance":    round(float(dist), 4),
        })
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_documents() -> list[str]:
    return sorted({m["document"] for m in _meta})


def get_all_chunks_for_document(document_name: str) -> list[str]:
    """Return all stored chunk texts for a given document name."""
    return [
        _chunks[i]
        for i, m in enumerate(_meta)
        if m["document"] == document_name
    ]


def delete_document(document_name: str) -> int:
    before = len(_chunks)
    _remove_document_chunks(document_name)
    removed = before - len(_chunks)
    if removed:
        _save_index()
    return removed


def get_db_stats() -> dict:
    if _faiss_index is None:
        return {"status": "not_initialised", "total_chunks": 0, "documents": []}
    return {
        "status":       "ready",
        "total_chunks": _faiss_index.ntotal,
        "documents":    list_documents(),
    }
