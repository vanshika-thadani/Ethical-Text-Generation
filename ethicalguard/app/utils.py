"""
utils.py — Shared utility helpers for EthicalGuard.

Keeps small, reusable functions out of the larger modules so they stay focused.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise raw extracted text before chunking or embedding.

    Steps:
      1. Unicode normalisation (NFKC) — collapses ligatures, fancy quotes, etc.
      2. Replace non-breaking spaces and other whitespace variants with a plain space.
      3. Collapse runs of whitespace into a single space.
      4. Strip leading / trailing whitespace.
    """
    # Normalise unicode (e.g. \u2019 → ', \u00a0 → space)
    text = unicodedata.normalize("NFKC", text)
    # Replace any whitespace character (tab, newline, etc.) with a space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split a long document into overlapping word-level chunks.

    Why overlapping chunks?
      If a sentence spans the boundary between two chunks, a non-overlapping
      split would cut it in half and lose context.  Overlap ensures that
      boundary sentences appear in full in at least one chunk, so retrieval
      never misses them.

    Parameters
    ----------
    text       : cleaned document text
    chunk_size : target number of words per chunk (default 400)
    overlap    : number of words shared between consecutive chunks (default 80)

    Returns
    -------
    List of non-empty string chunks.

    Example
    -------
    chunk_text("a b c d e", chunk_size=3, overlap=1)
    → ["a b c", "c d e"]
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` words of the current chunk.
        start += chunk_size - overlap

    return chunks


def truncate_text(text: str, max_words: int = 200) -> str:
    """
    Truncate text to at most max_words words, appending '...' if cut.

    Used when displaying chunks in API responses to keep payloads readable.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."
