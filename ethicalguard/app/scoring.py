"""
scoring.py — All safety and quality scoring logic for EthicalGuard.

Architecture (torch-free, ~50MB RAM)
--------------------------------------
  Toxicity, Sentiment, Bias  → HF Inference API
  Coherence (SBERT)          → HF Inference API (sentence-transformers/all-MiniLM-L6-v2)
  Fluency                    → constant 0.5 (distilgpt2 removed, torch eliminated)
  Manipulation penalty       → string matching (no model)

  torch / transformers / sentence-transformers: NOT imported.
  Total server RAM: ~50MB ✅
"""

from __future__ import annotations

import math
import os
import time
import requests as _http
import logging as _logging

from app.config import (
    WEIGHT_TOXICITY, WEIGHT_SENTIMENT, WEIGHT_BIAS, WEIGHT_COHERENCE,
    MANIPULATION_TRIGGERS, MANIPULATION_PENALTY_PER_WORD, MANIPULATION_PENALTY_MAX,
    SBERT_MODEL, EMPTY_OUTPUT_FALLBACK,
    TOXICITY_MODEL, SENTIMENT_MODEL, BIAS_MODEL,
)
from app.models import CandidateScores

_logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_hf_api_key: str = ""
# HF Inference API v2 — router subdomain is accessible where api-inference is blocked
_HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

# Kept for backwards-compat (rag.py calls set_rag_sbert_model which is now a no-op)
_sbert_model = None
_gen_tokenizer = None
_gen_model = None


# ---------------------------------------------------------------------------
# NaN / Inf safety helper
# ---------------------------------------------------------------------------

def safe_float(value: object, default: float = 0.0, label: str = "") -> float:
    """Return a JSON-safe float, substituting `default` for None/NaN/Inf."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        _logger.warning(f"safe_float: cannot convert {label!r}={value!r} → {default}")
        return default
    if math.isnan(v) or math.isinf(v):
        _logger.warning(f"safe_float: {label!r} is NaN/Inf → {default}")
        return default
    return round(v, 4)


# ---------------------------------------------------------------------------
# Startup injection
# ---------------------------------------------------------------------------

def set_scoring_models(hf_api_key: str, gen_tokenizer=None, gen_model=None):
    """
    Called once at startup by generation.py.
    Stores the HF API key — no local models loaded.
    gen_tokenizer / gen_model params kept for backwards-compat but ignored.
    """
    global _hf_api_key
    _hf_api_key = hf_api_key
    _logger.info(
        "Scoring ready: Toxicity/Sentiment/Bias/Coherence → HF Inference API | "
        "Fluency → constant 0.5"
    )


# ---------------------------------------------------------------------------
# HF Inference API helper
# ---------------------------------------------------------------------------

def _hf_post(endpoint: str, payload: dict, retries: int = 3) -> list:
    """
    POST to a HF Inference API endpoint.
    Handles 503 cold-start with exponential backoff.
    Returns empty list on persistent failure (graceful degradation).
    """
    headers = {"Authorization": f"Bearer {_hf_api_key}"}
    for attempt in range(retries):
        try:
            resp = _http.post(endpoint, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                wait = 10 * (attempt + 1)
                _logger.warning(f"HF model loading at {endpoint}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            result = resp.json()
            # Classification models return [[{label, score}, ...]] — unwrap outer list.
            # Feature-extraction (embeddings) returns [[float, ...], [float, ...]] —
            # do NOT unwrap, we need all vectors.
            if (isinstance(result, list) and result
                    and isinstance(result[0], list)
                    and result[0] and isinstance(result[0][0], dict)):
                return result[0]
            return result
        except Exception as exc:
            _logger.warning(f"HF API attempt {attempt + 1} failed ({endpoint}): {exc}")
            if attempt < retries - 1:
                time.sleep(3)
    _logger.error(f"HF API failed after {retries} attempts: {endpoint}")
    return []


def _hf_classify(text: str, model_id: str) -> list:
    """Text classification via HF Inference API."""
    return _hf_post(
        f"{_HF_API_BASE}/{model_id}",
        {"inputs": text[:512]}
    )


def _hf_embed(texts: list[str]) -> list[list[float]]:
    """
    Get sentence embeddings via HF Inference API (feature-extraction).
    Returns a list of 384-dim float vectors, one per input text.
    Falls back to zero vectors on failure.
    """
    result = _hf_post(
        f"{_HF_API_BASE}/{SBERT_MODEL}/pipeline/feature-extraction",
        {"inputs": texts}
    )
    if not result:
        # Return zero vectors as fallback
        return [[0.0] * 384 for _ in texts]
    # HF feature-extraction /pipeline/feature-extraction returns shape [n_texts, 384]
    # but for single input it may return just [384] (a flat list of floats).
    # Normalise to always be a list of lists.
    if result and isinstance(result[0], float):
        # Single text — wrap into a list of one vector
        result = [result]
    if isinstance(result[0], list) and isinstance(result[0][0], list):
        # Shape is [n_texts, seq_len, 384] — mean-pool over seq_len
        pooled = []
        for seq in result:
            dim = len(seq[0])
            mean_vec = [sum(seq[j][d] for j in range(len(seq))) / len(seq) for d in range(dim)]
            pooled.append(mean_vec)
        return pooled
    return result


# ---------------------------------------------------------------------------
# Safe label helpers
# ---------------------------------------------------------------------------
_SAFE_LABEL_KEYWORDS = {"nothate", "non-hate", "not hate", "normal", "neutral", "not_hate"}


def _pick_safe_score(results: list) -> float:
    """Return the score for the safe/non-toxic label from HF classification output."""
    for item in results:
        label_lower = item["label"].lower().replace("_", " ").replace("-", " ")
        if any(kw in label_lower for kw in _SAFE_LABEL_KEYWORDS):
            return round(item["score"], 4)
    _logger.warning(f"Could not identify safe label in: {results}")
    return 0.5


# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------

def _toxicity_score(text: str) -> float:
    """Toxicity safety score via HF API. 1.0=safe, 0.0=toxic."""
    results = _hf_classify(text, TOXICITY_MODEL)
    return _pick_safe_score(results) if results else 0.5


def _sentiment_score(text: str) -> float:
    """Sentiment quality score via HF API. Positive→1.0, Neutral→0.8, Negative→0.5-0.75."""
    results = _hf_classify(text, SENTIMENT_MODEL)
    if not results:
        return 0.8
    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()
    conf = best["score"]
    if "negative" in label:
        return round(1.0 - conf * 0.5, 4)
    elif "positive" in label:
        return round(min(1.0, 0.8 + conf * 0.2), 4)
    return 0.8


def _bias_score(text: str) -> float:
    """Bias safety score via HF API. 1.0=unbiased, 0.0=strongly biased."""
    results = _hf_classify(text, BIAS_MODEL)
    if not results:
        return 0.5
    best = max(results, key=lambda x: x["score"])
    if "biased" in best["label"].lower():
        return round(1.0 - best["score"], 4)
    return round(best["score"], 4)


def _coherence_score(prompt: str, response: str) -> float:
    """
    Cosine similarity between prompt and response embeddings via HF API.
    Uses sentence-transformers/all-MiniLM-L6-v2 through feature-extraction endpoint.
    """
    if not prompt or not prompt.strip() or not response or not response.strip():
        return 0.0

    try:
        vecs = _hf_embed([prompt, response])
        p, r = vecs[0], vecs[1]

        # Cosine similarity
        dot = sum(a * b for a, b in zip(p, r))
        norm_p = math.sqrt(sum(a * a for a in p))
        norm_r = math.sqrt(sum(b * b for b in r))
        if norm_p == 0 or norm_r == 0:
            return 0.0
        sim = dot / (norm_p * norm_r)
        return round(max(0.0, min(sim, 1.0)), 4)
    except Exception as exc:
        _logger.warning(f"_coherence_score failed: {exc}")
        return 0.0


def _fluency_score(_text: str) -> float:
    """
    Returns constant 0.5 (neutral).
    distilgpt2 removed to eliminate torch dependency and reduce RAM to ~50MB.
    Fluency is the least impactful scoring dimension — constant neutral is acceptable.
    """
    return 0.5


def _manipulation_penalty(text: str) -> float:
    """Scan for manipulation trigger phrases. Returns penalty in [0, 0.5]."""
    text_lower = text.lower()
    penalty = sum(
        MANIPULATION_PENALTY_PER_WORD
        for trigger in MANIPULATION_TRIGGERS
        if trigger in text_lower
    )
    return round(min(penalty, MANIPULATION_PENALTY_MAX), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(prompt: str, text: str, alpha: float) -> CandidateScores:
    """Compute all safety and quality metrics for a single candidate text."""
    if not text or not text.strip():
        _logger.warning("score_candidate(): empty text — substituting fallback.")
        text = EMPTY_OUTPUT_FALLBACK

    tox   = safe_float(_toxicity_score(text),         label="toxicity_score")
    sent  = safe_float(_sentiment_score(text),         label="sentiment_score")
    bias  = safe_float(_bias_score(text),              label="bias_score")
    coh   = safe_float(_coherence_score(prompt, text), label="coherence_score")
    flu   = 0.5   # constant — distilgpt2 removed
    manip = safe_float(_manipulation_penalty(text),    label="manipulation_penalty")

    ethics = safe_float(
        WEIGHT_TOXICITY * tox + WEIGHT_SENTIMENT * sent +
        WEIGHT_BIAS * bias + WEIGHT_COHERENCE * coh,
        label="ethics_score",
    )
    final = safe_float(
        alpha * ethics + (1 - alpha) * flu - manip,
        label="final_score",
    )

    return CandidateScores(
        text=text,
        toxicity_score=tox,
        sentiment_score=sent,
        bias_score=bias,
        coherence_score=coh,
        ethics_score=ethics,
        fluency_score=flu,
        manipulation_penalty=manip,
        final_score=final,
    )


def score_prompt_risk(prompt: str) -> float:
    """Toxicity risk of the prompt itself. 0.0=safe, 1.0=toxic."""
    return round(1.0 - _toxicity_score(prompt), 4)


def get_manipulation_penalty(text: str) -> float:
    """Public wrapper for _manipulation_penalty()."""
    return _manipulation_penalty(text)


# ---------------------------------------------------------------------------
# Batch scoring helpers (used by /analyze-document)
# ---------------------------------------------------------------------------

def batch_toxicity_scores(texts: list) -> list:
    """Toxicity scores for a list of texts via HF API."""
    return [safe_float(_toxicity_score(t), label="batch_tox") for t in texts]


def batch_bias_scores(texts: list) -> list:
    """Bias scores for a list of texts via HF API."""
    return [safe_float(_bias_score(t), label="batch_bias") for t in texts]
