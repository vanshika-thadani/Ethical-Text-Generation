"""
models.py — Pydantic request / response schemas for EthicalGuard API.

Keeping schemas in a dedicated file makes it easy to version the API
and reuse types across endpoints without circular imports.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List
from app.config import DEFAULT_MAX_TOKENS, DEFAULT_BEAMS, DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """
    Body for POST /generate.

    - text   : the user's prompt / incomplete sentence
    - max_tokens : how many new tokens the model may produce per candidate
    - beams  : number of independent candidates to generate and rerank
    - alpha  : weight balancing ethics (1.0) vs fluency (0.0)
    """
    text: str = Field(..., min_length=3, description="Input prompt")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class CompareRequest(BaseModel):
    """
    Body for POST /compare.

    Same fields as GenerateRequest — runs both a raw baseline generation
    and the full ethical reranking pipeline, then returns both side-by-side.
    """
    text: str = Field(..., min_length=3, description="Input prompt")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Shared score block (reused in both /generate and /compare responses)
# ---------------------------------------------------------------------------

class CandidateScores(BaseModel):
    """
    Full score breakdown for a single text candidate.

    Every field is computed once in scoring.score_candidate() and
    stored here so we never recompute the same score twice.
    """
    text: str
    toxicity_score: float       # 1.0 = safe, 0.0 = maximally toxic
    sentiment_score: float      # 1.0 = positive/neutral, lower = negative
    bias_score: float           # 1.0 = unbiased, 0.0 = strongly biased
    coherence_score: float      # cosine similarity between prompt & response
    ethics_score: float         # weighted composite of the four above
    fluency_score: float        # perplexity-based; 1.0 = very fluent
    manipulation_penalty: float # subtracted from final score
    final_score: float          # alpha*ethics + (1-alpha)*fluency - penalty


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class GenerateResponse(BaseModel):
    generated_text: str
    best_candidate: CandidateScores
    all_candidates: List[CandidateScores]


class BlockedResponse(BaseModel):
    """Returned when the prompt itself is flagged as unsafe."""
    status: str = "blocked"
    reason: str
    prompt_risk: float


class ImprovementMetrics(BaseModel):
    # toxicity_score and bias_score are SAFETY scores (higher = safer/less biased).
    # So best - baseline measures how much SAFER the ranked output is — a gain, not a reduction.
    toxicity_safety_gain: float  # positive = ranked output is safer (less toxic) than baseline
    bias_safety_gain: float      # positive = ranked output is less biased than baseline
    final_score_gain: float      # positive = ranked output scored higher overall


class CompareResponse(BaseModel):
    prompt: str
    prompt_risk: float          # toxicity risk of the prompt itself (0=safe, 1=harmful)
    baseline_output: str
    safety_ranked_output: str
    baseline_scores: CandidateScores
    safety_ranked_scores: CandidateScores
    improvement: ImprovementMetrics
