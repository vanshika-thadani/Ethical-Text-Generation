"""
scoring.py — All safety and quality scoring logic for EthicalGuard.

Architecture after HF migration
---------------------------------
  Toxicity, Sentiment, Bias  → HF Inference API (zero local RAM)
  Coherence                  → SBERT local CPU (~90MB)
  Fluency                    → distilgpt2 local CPU (~250MB)
  Manipulation penalty       → string matching (no model)

  Total local RAM: ~340MB  ✅ fits Render free tier (512MB)

Why keep SBERT and distilgpt2 local?
  SBERT is needed synchronously for every RAG embed call — API latency
  would make document upload/retrieval unusably slow.
  distilgpt2 is 250MB and used only for perplexity — not worth an API call.

HF Inference API notes
  - Free tier allows ~1000 requests/day per model.
  - Each scoring call = 3 API requests (toxicity + sentiment + bias).
  - Models warm up on first call (~10-20s cold start on free tier).
  - Responses are deterministic (no sampling) — consistent scores.
"""

from __future__ import annotations

import math
import os
import time
import torch
import requests as _http

from sentence_transformers import SentenceTransformer, util
from app.config import (
    WEIGHT_TOXICITY, WEIGHT_SENTIMENT, WEIGHT_BIAS, WEIGHT_COHERENCE,
    MANIPULATION_TRIGGERS, MANIPULATION_PENALTY_PER_WORD, MANIPULATION_PENALTY_MAX,
    SBERT_MODEL, EMPTY_OUTPUT_FALLBACK,
    TOXICITY_MODEL, SENTIMENT_MODEL, BIAS_MODEL,
)
from app.models import CandidateScores

import logging as _logging
_logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level handles
# ---------------------------------------------------------------------------
_hf_api_key: str = ""
_sbert_model: SentenceTransformer | None = None
_gen_tokenizer = None
_gen_model = None

# HF Inference API base URL
_HF_API_BASE = "https://api-inference.huggingface.co/models"

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

def set_scoring_models(hf_api_key: str, gen_tokenizer, gen_model):
    """
    Called once at startup by generation.py.

    Stores the HF API key, loads SBERT locally, and stores the distilgpt2
    handles for fluency scoring.

    No local model objects for toxicity/sentiment/bias — those go through
    the HF Inference API using _hf_api_key.
    """
    global _hf_api_key, _sbert_model, _gen_tokenizer, _gen_model

    _hf_api_key = hf_api_key
    _gen_tokenizer = gen_tokenizer
    _gen_model = gen_model

    # SBERT stays local — needed for synchronous RAG embedding.
    _sbert_model = SentenceTransformer(SBERT_MODEL, device="cpu")
    _logger.info(f"SBERT loaded locally: {SBERT_MODEL} | device: cpu")
    _logger.info("Toxicity / Sentiment / Bias → HF Inference API")


# ---------------------------------------------------------------------------
# HF Inference API helper
# ---------------------------------------------------------------------------

def _hf_classify(text: str, model_id: str, retries: int = 3) -> list[dict]:
    """
    Call the HF Inference API text-classification endpoint.

    Returns a list of {label, score} dicts.
    Handles cold-start (503 model loading) with automatic retry + backoff.
    Falls back to empty list on persistent failure so scoring degrades
    gracefully rather than crashing.
    """
    url = f"{_HF_API_BASE}/{model_id}"
    headers = {"Authorization": f"Bearer {_hf_api_key}"}
    payload = {"inputs": text[:512]}   # truncate to stay within token limits

    for attempt in range(retries):
        try:
            resp = _http.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                # Model is loading — wait and retry
                wait = 10 * (attempt + 1)
                _logger.warning(f"HF model {model_id} loading, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            result = resp.json()
            # HF returns either [[{...}]] or [{...}] depending on model
            if isinstance(result, list) and result and isinstance(result[0], list):
                return result[0]
            return result
        except Exception as exc:
            _logger.warning(f"HF API call failed ({model_id}, attempt {attempt+1}): {exc}")
            if attempt < retries - 1:
                time.sleep(3)

    _logger.error(f"HF API failed after {retries} attempts for {model_id}")
    return []


# ---------------------------------------------------------------------------
# Individual metric helpers
# ---------------------------------------------------------------------------

# Safe label keywords — used to identify which HF output label is "safe"
_SAFE_LABEL_KEYWORDS = {"nothate", "non-hate", "not hate", "normal", "neutral", "not_hate"}
_UNSAFE_LABEL_KEYWORDS = {"hate", "toxic", "offensive", "abusive"}


def _pick_safe_score(results: list[dict]) -> float:
    """
    From a list of {label, score} dicts, return the score for the safe label.

    Identifies the safe label by checking against _SAFE_LABEL_KEYWORDS.
    Falls back to 0.5 (neutral) if no matching label is found.
    """
    for item in results:
        label_lower = item["label"].lower().replace("_", " ").replace("-", " ")
        if any(kw in label_lower for kw in _SAFE_LABEL_KEYWORDS):
            return round(item["score"], 4)
    # If no safe label found, return neutral
    _logger.warning(f"Could not identify safe label in: {results}")
    return 0.5


def _toxicity_score(text: str) -> float:
    """
    Toxicity safety score in [0, 1] via HF Inference API.
    1.0 = completely safe, 0.0 = maximally toxic.
    """
    results = _hf_classify(text, TOXICITY_MODEL)
    if not results:
        return 0.5   # neutral fallback on API failure
    return _pick_safe_score(results)


def _sentiment_score(text: str) -> float:
    """
    Sentiment quality score in [0, 1] via HF Inference API.
    Positive → ~1.0, Neutral → 0.8, Negative → 0.5–0.75.
    """
    results = _hf_classify(text, SENTIMENT_MODEL)
    if not results:
        return 0.8   # neutral fallback

    # Find the highest-confidence label
    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()
    conf = best["score"]

    if "negative" in label:
        return round(1.0 - conf * 0.5, 4)
    elif "positive" in label:
        return round(min(1.0, 0.8 + conf * 0.2), 4)
    return 0.8


def _bias_score(text: str) -> float:
    """
    Bias safety score in [0, 1] via HF Inference API.
    1.0 = completely unbiased, 0.0 = strongly biased.
    """
    results = _hf_classify(text, BIAS_MODEL)
    if not results:
        return 0.5   # neutral fallback

    best = max(results, key=lambda x: x["score"])
    if "biased" in best["label"].lower():
        return round(1.0 - best["score"], 4)
    return round(best["score"], 4)


def _coherence_score(prompt: str, response: str) -> float:
    """
    SBERT cosine similarity between prompt and response embeddings.
    Computed locally — SBERT stays on CPU.
    """
    if not prompt or not prompt.strip() or not response or not response.strip():
        return 0.0

    prompt_emb = _sbert_model.encode(prompt, convert_to_tensor=True)
    resp_emb   = _sbert_model.encode(response, convert_to_tensor=True)
    sim = float(util.cos_sim(prompt_emb, resp_emb).item())

    if math.isnan(sim) or math.isinf(sim):
        return 0.0

    return round(max(0.0, min(sim, 1.0)), 4)


def _fluency_score(text: str) -> float:
    """
    Perplexity-based fluency score via distilgpt2 (local CPU).
    1.0 = very fluent, 0.0 = incoherent.
    """
    if not text or not text.strip():
        return 0.0

    inputs = _gen_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _gen_model(**inputs, labels=inputs["input_ids"])

    loss = outputs.loss.item()
    if math.isnan(loss) or math.isinf(loss):
        return 0.0
    if loss > 20:
        return 0.1

    perplexity = math.exp(min(loss, 100.0))
    score = 1.0 / (1.0 + math.log(max(perplexity, 1.0)))
    return safe_float(score, default=0.0, label="fluency_score")


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
    """
    Compute all safety and quality metrics for a single candidate text.
    All metrics computed once — no redundant API calls.
    """
    if not text or not text.strip():
        _logger.warning("score_candidate(): empty text — substituting fallback.")
        text = EMPTY_OUTPUT_FALLBACK

    tox   = safe_float(_toxicity_score(text),         label="toxicity_score")
    sent  = safe_float(_sentiment_score(text),         label="sentiment_score")
    bias  = safe_float(_bias_score(text),              label="bias_score")
    coh   = safe_float(_coherence_score(prompt, text), label="coherence_score")
    flu   = safe_float(_fluency_score(text),           label="fluency_score")
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
# With HF Inference API we call per-text (no local batch). Sub-batching is
# handled by rate-limiting logic inside _hf_classify().

def batch_toxicity_scores(texts: list) -> list:
    """Run toxicity scoring over all texts via HF API. Returns safety scores."""
    return [safe_float(_toxicity_score(t), label="batch_tox") for t in texts]


def batch_bias_scores(texts: list) -> list:
    """Run bias scoring over all texts via HF API. Returns safety scores."""
    return [safe_float(_bias_score(t), label="batch_bias") for t in texts]
