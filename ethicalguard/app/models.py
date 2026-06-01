"""
models.py — Pydantic request / response schemas for EthicalGuard API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List
from app.config import DEFAULT_MAX_TOKENS, REWRITE_MAX_TOKENS, DEFAULT_BEAMS, DEFAULT_ALPHA


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Input prompt")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Input prompt")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Shared score block
# ---------------------------------------------------------------------------

class CandidateScores(BaseModel):
    text: str
    toxicity_score: float
    sentiment_score: float
    bias_score: float
    coherence_score: float
    ethics_score: float
    fluency_score: float
    manipulation_penalty: float
    final_score: float


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class GenerateResponse(BaseModel):
    generated_text: str
    best_candidate: CandidateScores
    all_candidates: List[CandidateScores]


class BlockedResponse(BaseModel):
    status: str = "blocked"
    reason: str
    prompt_risk: float


class ImprovementMetrics(BaseModel):
    toxicity_safety_gain: float
    bias_safety_gain: float
    final_score_gain: float


class CompareResponse(BaseModel):
    prompt: str
    prompt_risk: float
    baseline_output: str
    safety_ranked_output: str
    baseline_scores: CandidateScores
    safety_ranked_scores: CandidateScores
    improvement: ImprovementMetrics


# ---------------------------------------------------------------------------
# RAG schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    status: str = "success"
    document_name: str
    chunks_added: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)
    document_name: Optional[str] = Field(default=None)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    text: str
    document: str
    chunk_index: int
    distance: float


class AskResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    ethical_scores: CandidateScores


class ChunkAnalysis(BaseModel):
    chunk: str
    chunk_index: int
    toxicity_score: float
    toxicity_risk: float
    bias_score: float
    manipulation_penalty: float
    ethics_score: float
    flagged: bool
    severity: str = "LOW"


class AnalyzeDocumentRequest(BaseModel):
    document_name: str = Field(..., description="Name of the previously uploaded document")


class AnalyzeDocumentResponse(BaseModel):
    document_name: str
    total_chunks: int
    flagged_chunks: int
    unsafe_chunks: List[ChunkAnalysis]
    all_chunks: List[ChunkAnalysis]


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=3)
    # Default to REWRITE_MAX_TOKENS (60) — much lower than DEFAULT_MAX_TOKENS (100)
    # so Phi-3 doesn't have room to ramble. User can still override up to 200.
    max_tokens: int = Field(default=REWRITE_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class RewriteResponse(BaseModel):
    original: str
    ethical_rewrite: str
    scores_before: CandidateScores
    scores_after: CandidateScores


class RAGStatusResponse(BaseModel):
    status: str
    total_chunks: int
    documents: List[str]