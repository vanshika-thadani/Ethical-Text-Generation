"""
main.py — FastAPI application entry point for EthicalGuard.

Endpoints
---------
GET  /          → health check
POST /generate  → ethical reranked text generation
POST /compare   → baseline vs safety-ranked comparison

Startup sequence
----------------
All heavy models are loaded once during the FastAPI lifespan event so the
first request is never slow due to model initialisation.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import generation, scoring
from app.config import PROMPT_BLOCK_THRESHOLD
from app.models import (
    GenerateRequest,
    CompareRequest,
    GenerateResponse,
    CompareResponse,
    BlockedResponse,
    CandidateScores,
    ImprovementMetrics,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; after `yield` at shutdown.
    """
    logger.info("EthicalGuard starting — loading models...")
    try:
        loaded = generation.load_models()
        logger.info(f"Ready. Generation model: {loaded}")
    except RuntimeError as exc:
        # Log the error but let the app start so /health still responds.
        logger.error(f"Model loading failed: {exc}")
    yield
    logger.info("EthicalGuard shutting down.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EthicalGuard: Safety-Aware LLM Text Generation",
    description=(
        "Generates multiple candidate completions and selects the most ethical "
        "one using multi-dimensional safety scoring (toxicity, sentiment, bias, "
        "coherence, fluency).  Includes prompt safety blocking and a "
        "baseline-vs-ethical comparison endpoint."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: check models are ready
# ---------------------------------------------------------------------------

def _require_models():
    """Raise a clear 503 if models haven't loaded yet."""
    if generation.gen_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are not loaded. The server may still be initialising "
                "or model loading failed. Check server logs for details."
            ),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the server is running."""
    return {
        "status": "running",
        "service": "EthicalGuard",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.post(
    "/generate",
    response_model=GenerateResponse,
    tags=["Generation"],
    summary="Generate ethically reranked text",
)
def generate(req: GenerateRequest):
    """
    Core endpoint: generate N candidates and return the safest one.

    Pipeline
    --------
    1. Check prompt safety — block if toxicity risk > threshold.
    2. Generate `beams` independent completions.
    3. Score every candidate with the full safety pipeline.
    4. Return the candidate with the highest final_score.

    The response includes the best candidate's full score breakdown AND
    all candidates so clients can inspect the reranking decision.
    """
    _require_models()

    # ---- Step 1: Prompt safety gate ----
    # We evaluate the prompt itself before spending compute on generation.
    # If the user's input is already harmful, we refuse immediately.
    prompt_risk = scoring.score_prompt_risk(req.text)
    if prompt_risk > PROMPT_BLOCK_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "blocked",
                "reason": "Prompt contains potentially unsafe or harmful content",
                "prompt_risk": prompt_risk,
            },
        )

    # ---- Step 2: Generate candidates ----
    try:
        candidates_text = generation.generate_candidates(
            req.text, req.beams, req.max_tokens
        )
    except Exception as exc:
        logger.error(f"/generate — generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}")

    # ---- Step 3: Score every candidate (once each) ----
    try:
        scored: list[CandidateScores] = [
            scoring.score_candidate(req.text, text, req.alpha)
            for text in candidates_text
        ]
    except Exception as exc:
        logger.error(f"/generate — scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {exc}")

    # ---- Step 4: Select best ----
    best = max(scored, key=lambda c: c.final_score)

    return GenerateResponse(
        generated_text=best.text,
        best_candidate=best,
        all_candidates=scored,
    )


@app.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Evaluation"],
    summary="Compare raw baseline vs ethical reranked output",
)
def compare(req: CompareRequest):
    """
    Side-by-side comparison of raw LLM output vs ethically reranked output.

    Baseline
    --------
    A single generation with NO reranking — whatever the model produces first.
    This represents what a naive LLM deployment would return.

    Safety-ranked
    -------------
    The best candidate selected from `beams` generations using the full
    ethical scoring pipeline.

    The `improvement` block quantifies how much safer the reranked output is:
      - toxicity_reduction  : positive = less toxic
      - bias_reduction      : positive = less biased
      - final_score_gain    : positive = higher overall quality

    This endpoint is designed for research demonstrations, benchmarks, and
    explaining the value of ethical reranking to stakeholders.

    Unlike /generate, unsafe prompts are NOT blocked here. The prompt_risk
    score is included in the response so researchers can study how the
    reranking pipeline handles harmful inputs.
    """
    _require_models()

    # ---- Assess prompt risk (informational only — no blocking) ----
    # /compare is a research/benchmark endpoint. We always proceed with
    # generation and surface the risk score so it can be studied.
    prompt_risk = scoring.score_prompt_risk(req.text)

    # ---- Baseline: single raw generation, no reranking ----
    try:
        baseline_text = generation.generate_one(req.text, req.max_tokens)
    except Exception as exc:
        logger.error(f"/compare — baseline generation failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Baseline generation failed: {exc}"
        )

    # ---- Safety-ranked: generate N candidates, pick best ----
    try:
        candidates_text = generation.generate_candidates(
            req.text, req.beams, req.max_tokens
        )
        scored: list[CandidateScores] = [
            scoring.score_candidate(req.text, text, req.alpha)
            for text in candidates_text
        ]
        best = max(scored, key=lambda c: c.final_score)
    except Exception as exc:
        logger.error(f"/compare — safety-ranked generation failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Safety-ranked generation failed: {exc}"
        )

    # ---- Score the baseline (for fair comparison) ----
    try:
        baseline_scores = scoring.score_candidate(req.text, baseline_text, req.alpha)
    except Exception as exc:
        logger.error(f"/compare — baseline scoring failed: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Baseline scoring failed: {exc}"
        )

    # ---- Compute improvement metrics ----
    # toxicity_score and bias_score are SAFETY scores (1.0 = safe/unbiased).
    # best - baseline therefore measures safety GAIN: positive = ranked output is safer.
    improvement = ImprovementMetrics(
        toxicity_safety_gain=round(
            best.toxicity_score - baseline_scores.toxicity_score, 4
        ),
        bias_safety_gain=round(
            best.bias_score - baseline_scores.bias_score, 4
        ),
        final_score_gain=round(
            best.final_score - baseline_scores.final_score, 4
        ),
    )

    return CompareResponse(
        prompt=req.text,
        prompt_risk=prompt_risk,
        baseline_output=baseline_text,
        safety_ranked_output=best.text,
        baseline_scores=baseline_scores,
        safety_ranked_scores=best,
        improvement=improvement,
    )
