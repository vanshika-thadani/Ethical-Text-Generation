"""
ingestion.py — Document ingestion pipeline for EthicalGuard RAG.

Responsibilities
----------------
1. Accept an uploaded file (txt or pdf).
2. Extract raw text from the file.
3. Clean and chunk the text.
4. Return chunks ready for embedding and storage.

Supported formats
-----------------
.txt  — read directly as UTF-8 text.
.pdf  — extract text page-by-page using pdfplumber (preferred) with a
        PyPDF2 fallback so the server still works if one library is missing.

Why pdfplumber over PyPDF2?
  pdfplumber handles complex layouts (tables, columns) better and returns
  cleaner text.  PyPDF2 is kept as a fallback for compatibility.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.utils import clean_text, chunk_text
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_txt(file_path: str) -> str:
    """Read a plain-text file and return its contents as a string."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all readable text from a PDF file.

    Tries pdfplumber first (better layout handling), falls back to PyPDF2.
    Raises ValueError if neither library can extract any text.
    """
    text_pages: list[str] = []

    # ---- Attempt 1: pdfplumber ----
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        if text_pages:
            logger.info(f"pdfplumber extracted {len(text_pages)} pages from {file_path}")
            return "\n".join(text_pages)
    except ImportError:
        logger.warning("pdfplumber not installed — trying PyPDF2.")
    except Exception as exc:
        logger.warning(f"pdfplumber failed ({exc}) — trying PyPDF2.")

    # ---- Attempt 2: PyPDF2 fallback ----
    try:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
        if text_pages:
            logger.info(f"PyPDF2 extracted {len(text_pages)} pages from {file_path}")
            return "\n".join(text_pages)
    except ImportError:
        logger.warning("PyPDF2 not installed.")
    except Exception as exc:
        logger.warning(f"PyPDF2 failed: {exc}")

    raise ValueError(
        f"Could not extract text from PDF '{file_path}'. "
        "Install pdfplumber or PyPDF2 and ensure the PDF contains selectable text."
    )


def extract_text(file_path: str) -> str:
    """
    Dispatch to the correct extractor based on file extension.

    Raises
    ------
    ValueError  if the file type is not supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: .txt, .pdf"
        )


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_document(file_path: str) -> list[str]:
    """
    Full ingestion pipeline: extract → clean → chunk.

    Parameters
    ----------
    file_path : str
        Path to the uploaded file on disk.

    Returns
    -------
    List of text chunks ready to be embedded and stored in the vector DB.

    Raises
    ------
    ValueError  if extraction fails or the document is empty after cleaning.
    """
    # Step 1: Extract raw text
    raw_text = extract_text(file_path)
    if not raw_text.strip():
        raise ValueError("Document appears to be empty or contains no extractable text.")

    # Step 2: Clean (normalise unicode, collapse whitespace)
    cleaned = clean_text(raw_text)
    logger.info(f"Extracted {len(cleaned.split())} words from '{file_path}'")

    # Step 3: Chunk into overlapping segments
    # Chunking splits the document into pieces small enough to embed and
    # retrieve individually.  Overlap preserves context at chunk boundaries.
    chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    logger.info(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not chunks:
        raise ValueError("Document produced no chunks after processing.")

    return chunks
