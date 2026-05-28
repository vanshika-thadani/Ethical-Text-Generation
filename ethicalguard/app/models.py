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


# ---------------------------------------------------------------------------
# RAG request / response schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Returned after a successful document upload and ingestion."""
    status: str = "success"
    document_name: str
    chunks_added: int


class AskRequest(BaseModel):
    """
    Body for POST /ask.

    question      : the user's natural-language question about the document
    top_k         : how many chunks to retrieve from the vector DB
    document_name : optional — restrict retrieval to a specific document
    max_tokens    : tokens to generate for the answer
    alpha         : ethics/fluency weight for reranking the answer
    """
    question: str = Field(..., min_length=3, description="Question about the document")
    top_k: int = Field(default=3, ge=1, le=10)
    document_name: Optional[str] = Field(default=None, description="Filter by document name")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    """A single chunk returned by the retrieval step."""
    text: str
    document: str
    chunk_index: int
    distance: float   # cosine distance — lower means more relevant


class AskResponse(BaseModel):
    question: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    ethical_scores: CandidateScores


class ChunkAnalysis(BaseModel):
    """Ethical analysis result for a single document chunk."""
    chunk: str
    chunk_index: int
    toxicity_score: float       # safety score: 1.0 = safe, 0.0 = toxic
    toxicity_risk: float        # risk score: 1 - toxicity_score (high = dangerous)
    bias_score: float           # safety score: 1.0 = unbiased
    manipulation_penalty: float # 0.0–0.5, higher = more manipulative
    ethics_score: float         # composite safety score
    flagged: bool
    severity: str = "LOW"       # LOW / MEDIUM / HIGH


class AnalyzeDocumentRequest(BaseModel):
    """Body for POST /analyze-document."""
    document_name: str = Field(..., description="Name of the previously uploaded document")


class AnalyzeDocumentResponse(BaseModel):
    document_name: str
    total_chunks: int
    flagged_chunks: int
    unsafe_chunks: List[ChunkAnalysis]
    all_chunks: List[ChunkAnalysis]


class RewriteRequest(BaseModel):
    """Body for POST /rewrite."""
    text: str = Field(..., min_length=3, description="Text to rewrite ethically")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=10, le=200)
    beams: int = Field(default=DEFAULT_BEAMS, ge=1, le=10)
    alpha: float = Field(default=DEFAULT_ALPHA, ge=0.0, le=1.0)


class RewriteResponse(BaseModel):
    original: str
    ethical_rewrite: str
    scores_before: CandidateScores
    scores_after: CandidateScores


class RAGStatusResponse(BaseModel):
    """Returned by GET /rag-status."""
    status: str
    total_chunks: int
    documents: List[str]
