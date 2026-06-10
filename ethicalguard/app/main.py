"""
main.py — FastAPI application entry point for EthicalGuard.

Endpoints
---------
GET  /              → health check
GET  /rag-status    → vector DB stats
POST /generate      → ethical reranked text generation
POST /compare       → baseline vs safety-ranked comparison
POST /upload        → ingest a document (txt/pdf) into the vector DB
POST /ask           → RAG question answering over uploaded documents
POST /analyze-document → ethical analysis of every chunk in a document
POST /rewrite       → rewrite toxic/manipulative text into safer language

Startup sequence
----------------
All heavy models are loaded once during the FastAPI lifespan event.
ChromaDB is also initialised at startup so the first request is instant.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app import generation, scoring, rag, ingestion
from app.config import (
    PROMPT_BLOCK_THRESHOLD,
    UPLOAD_DIR,
    RAG_TOP_K,
    DEFAULT_ALPHA,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BEAMS,
)

from app.models import (
    # existing
    GenerateRequest, CompareRequest,
    GenerateResponse, CompareResponse,
    BlockedResponse, CandidateScores, ImprovementMetrics,
    # new RAG
    UploadResponse, AskRequest, AskResponse,
    RetrievedChunk, AnalyzeDocumentRequest, AnalyzeDocumentResponse,
    ChunkAnalysis, RewriteRequest, RewriteResponse, RAGStatusResponse,
    #extension
    ChunksInput
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
# Lifespan — load models + init vector DB once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs at startup; after `yield` at shutdown.
    """
    # Ensure upload and vectordb directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    logger.info("EthicalGuard starting — loading models...")
    try:
        loaded = generation.load_models()
        # load_models() also calls rag.set_rag_sbert_model() and rag.init_vector_db()
        logger.info(f"Ready. Generation model: {loaded}")
    except RuntimeError as exc:
        logger.error(f"Model loading failed: {exc}")
    yield
    logger.info("EthicalGuard shutting down.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EthicalGuard: Safety-Aware LLM + RAG System",
    description=(
        "Ethical reranking, document ingestion, RAG question answering, "
        "ethical document analysis, and safe text rewriting — all in one backend."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
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


def _require_rag():
    """Raise a clear 503 if the vector DB isn't ready."""
    stats = rag.get_db_stats()
    if stats["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="Vector DB is not initialised. Check server logs.",
        )


# ---------------------------------------------------------------------------
# ── EXISTING ENDPOINTS (unchanged) ─────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the server is running."""
    return {
        "status": "running",
        "service": "EthicalGuard",
        "version": "3.0.0",
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
    Generate N candidates and return the safest one.

    Pipeline: prompt safety gate → generate N → score all → return best.
    """
    _require_models()

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

    try:
        candidates_text = generation.generate_candidates(req.text, req.beams, req.max_tokens)
    except Exception as exc:
        logger.error(f"/generate — generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}")

    try:
        scored: List[CandidateScores] = [
            scoring.score_candidate(req.text, text, req.alpha)
            for text in candidates_text
        ]
    except Exception as exc:
        logger.error(f"/generate — scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Scoring pipeline failed: {exc}")

    best = max(scored, key=lambda c: c.final_score)
    return GenerateResponse(generated_text=best.text, best_candidate=best, all_candidates=scored)


@app.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Evaluation"],
    summary="Compare raw baseline vs ethical reranked output",
)
def compare(req: CompareRequest):
    """
    Side-by-side comparison of raw LLM output vs ethically reranked output.
    Unsafe prompts are NOT blocked — prompt_risk is included in the response.
    """
    _require_models()

    prompt_risk = scoring.score_prompt_risk(req.text)

    try:
        baseline_text = generation.generate_one(req.text, req.max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline generation failed: {exc}")

    try:
        candidates_text = generation.generate_candidates(req.text, req.beams, req.max_tokens)
        scored: List[CandidateScores] = [
            scoring.score_candidate(req.text, text, req.alpha)
            for text in candidates_text
        ]
        best = max(scored, key=lambda c: c.final_score)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Safety-ranked generation failed: {exc}")

    try:
        baseline_scores = scoring.score_candidate(req.text, baseline_text, req.alpha)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Baseline scoring failed: {exc}")

    improvement = ImprovementMetrics(
        toxicity_safety_gain=round(best.toxicity_score - baseline_scores.toxicity_score, 4),
        bias_safety_gain=round(best.bias_score - baseline_scores.bias_score, 4),
        final_score_gain=round(best.final_score - baseline_scores.final_score, 4),
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


# ---------------------------------------------------------------------------
# ── NEW RAG ENDPOINTS ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get(
    "/rag-status",
    response_model=RAGStatusResponse,
    tags=["RAG"],
    summary="Vector DB status and document list",
)
def rag_status():
    """
    Returns the current state of the vector DB:
    how many chunks are stored and which documents have been uploaded.
    """
    stats = rag.get_db_stats()
    return RAGStatusResponse(**stats)


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["RAG"],
    summary="Upload and ingest a document (txt or pdf)",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a .txt or .pdf file, extract its text, chunk it, embed the chunks,
    and store them in the vector DB.

    Pipeline
    --------
    1. Save the uploaded file to the uploads/ directory.
    2. Extract text (pdfplumber for PDF, direct read for txt).
    3. Clean and chunk the text (~400 words per chunk, 80-word overlap).
    4. Embed each chunk with SBERT (all-MiniLM-L6-v2).
    5. Upsert into ChromaDB (re-uploading the same file is safe — no duplicates).

    Supported formats: .txt, .pdf
    """
    _require_models()

    # Validate file extension
    allowed = {".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}",
        )

    # Save file to uploads/ with a unique prefix to avoid collisions
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Extract text and chunk
    try:
        chunks = ingestion.ingest_document(save_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"/upload — ingestion failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}")

    # Store in vector DB — use the original filename (without UUID prefix) as the document name
    document_name = file.filename
    try:
        added = rag.add_chunks_to_db(document_name, chunks)
    except Exception as exc:
        logger.error(f"/upload — vector DB write failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Vector DB storage failed: {exc}")

    return UploadResponse(
        status="success",
        document_name=document_name,
        chunks_added=added,
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Ask a question about uploaded documents",
)
def ask(req: AskRequest):
    """
    RAG question answering over uploaded documents.

    Pipeline
    --------
    1. Embed the question with SBERT.
    2. Retrieve the top-k most semantically relevant chunks from ChromaDB.
    3. Build a context string from the retrieved chunks.
    4. Generate an answer conditioned on the context using the LLM.
    5. Ethically rerank the answer using the full scoring pipeline.

    The retrieved_chunks field in the response lets you trace exactly which
    passages the answer was grounded in — full transparency.
    """
    _require_models()

    # Step 1 & 2: Retrieve relevant chunks
    try:
        chunks = rag.retrieve_chunks(
            question=req.question,
            top_k=req.top_k,
            document_name=req.document_name,
        )
    except Exception as exc:
        logger.error(f"/ask — retrieval failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant chunks found. "
                "Please upload a document first using POST /upload."
            ),
        )

    # Step 3: Build context string from retrieved chunks
    # We concatenate the top-k chunks separated by a divider so the LLM
    # can see all relevant passages at once.
    context = "\n---\n".join(c["text"] for c in chunks)

    # Step 4: Build a RAG prompt that grounds the answer in the retrieved context
    rag_prompt = (
        f"Based on the following document excerpts, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.question}\n"
        f"Answer:"
    )

    # Step 5: Generate and ethically rerank the answer
    try:
        candidates_text = generation.generate_candidates(rag_prompt, DEFAULT_BEAMS, req.max_tokens)
        scored: List[CandidateScores] = [
            # Score against the original question (not the full RAG prompt)
            # so coherence measures relevance to what the user actually asked.
            scoring.score_candidate(req.question, text, req.alpha)
            for text in candidates_text
        ]
        best = max(scored, key=lambda c: c.final_score)
    except Exception as exc:
        logger.error(f"/ask — generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")

    return AskResponse(
        question=req.question,
        retrieved_chunks=[RetrievedChunk(**c) for c in chunks],
        answer=best.text,
        ethical_scores=best,
    )


# FIND and REPLACE the entire analyze_document() function:

@app.post(
    "/analyze-document",
    response_model=AnalyzeDocumentResponse,
    tags=["RAG"],
    summary="Ethical analysis of every sentence in an uploaded document",
)
def analyze_document(req: AnalyzeDocumentRequest):
    """
    Score every sentence of an uploaded document using a two-stage pipeline.

    Stage 1 — Batch pre-filter (fast):
        Run toxicity + bias over ALL sentences in one batch call each.
        Compute manipulation penalty (string matching, near-instant).
        Sentences with tox_risk > 0.12, bias_risk > 0.12, or manip > 0
        are marked suspicious — threshold is wide to avoid missing subtle bias.

    Stage 2 — Full scoring (suspicious sentences only):
        Run complete score_candidate() on suspicious sentences only.
        Severity uses full.ethics_score < 0.65 as additional MEDIUM trigger.
        ~20-30% of sentences reach Stage 2 for a typical document.

    LOW sentences skip Stage 2 — ethics approximated from batch scores.
    Fluency skipped entirely — irrelevant for document safety analysis.

    Speed: 300 sentences ~20-30s vs ~90-150s with old per-sentence approach.
    """
    _require_models()

    from app.utils import split_into_sentences, clean_text
    from app.ingestion import extract_text

    # Find uploaded file (saved with UUID prefix)
    matched_path = None
    for fname in os.listdir(UPLOAD_DIR):
        if fname.endswith("_" + req.document_name) or fname == req.document_name:
            matched_path = os.path.join(UPLOAD_DIR, fname)
            break

    if not matched_path:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{req.document_name}' not found. Upload it first via POST /upload.",
        )

    try:
        raw_text = extract_text(matched_path)
        sentences = split_into_sentences(clean_text(raw_text))
    except Exception as exc:
        logger.error(f"/analyze-document — text extraction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {exc}")

    if not sentences:
        raise HTTPException(status_code=422, detail="Document produced no scorable sentences.")

    logger.info(f"/analyze-document — scoring {len(sentences)} sentences from '{req.document_name}'")

    # Stage 1: batch pre-filter
    try:
        tox_scores  = scoring.batch_toxicity_scores(sentences)
        bias_scores = scoring.batch_bias_scores(sentences)
    except Exception as exc:
        logger.error(f"/analyze-document — batch scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {exc}")

    manip_penalties = [scoring.get_manipulation_penalty(s) for s in sentences]

    chunk_analyses: List[ChunkAnalysis] = []

    for idx, sentence in enumerate(sentences):
        tox_score  = tox_scores[idx]
        bias_score = bias_scores[idx]
        tox_risk   = round(1.0 - tox_score, 4)
        bias_risk  = round(1.0 - bias_score, 4)
        manip      = manip_penalties[idx]

        # Stage 2: full scoring for suspicious sentences only
        suspicious = tox_risk > 0.12 or bias_risk > 0.12 or manip > 0

        if suspicious:
            full = scoring.score_candidate(sentence, sentence, DEFAULT_ALPHA)
            if tox_risk > 0.5 or bias_risk > 0.5 or manip > 0.3:
                severity = "HIGH"
            elif tox_risk > 0.2 or bias_risk > 0.2 or manip > 0.1 or full.ethics_score < 0.65:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            ethics_score = full.ethics_score
        else:
            # Approximate ethics for LOW sentences — no model forward pass
            ethics_score = round(
                0.40 * tox_score + 0.25 * 0.8 + 0.25 * bias_score + 0.10 * 1.0, 4
            )
            severity = "LOW"

        logger.info(
            f"Sentence {idx} | tox_risk={tox_risk:.3f} bias_risk={bias_risk:.3f} "
            f"manip={manip:.2f} suspicious={suspicious} severity={severity}"
        )

        chunk_analyses.append(ChunkAnalysis(
            chunk=sentence,
            chunk_index=idx,
            toxicity_score=tox_score,
            toxicity_risk=tox_risk,
            bias_score=bias_score,
            manipulation_penalty=manip,
            ethics_score=ethics_score,
            flagged=severity in ("HIGH", "MEDIUM"),
            severity=severity,
        ))

    unsafe = [c for c in chunk_analyses if c.flagged]

    return AnalyzeDocumentResponse(
        document_name=req.document_name,
        total_chunks=len(chunk_analyses),
        flagged_chunks=len(unsafe),
        unsafe_chunks=unsafe,
        all_chunks=chunk_analyses,
    )


@app.post(
    "/rewrite",
    response_model=RewriteResponse,
    tags=["RAG"],
    summary="Rewrite toxic or manipulative text into safer communication",
)
def rewrite(req: RewriteRequest):
    """
    Rewrite a piece of text into a safer, more ethical version.

    Pipeline
    --------
    1. Score the original text to establish a baseline.
    2. Generate N rewrite candidates using the ethical instruction template.
    3. Score all candidates and select the one with the highest final_score.
    4. Return original + rewrite + before/after scores for comparison.

    Use cases
    ---------
    - Rewriting a flagged document chunk identified by /analyze-document
    - Cleaning up manipulative marketing copy
    - Improving the tone of emotionally charged messages
    - Browser extension: rewrite webpage content on the fly (future)
    """
    _require_models()

    # Score the original text first
    try:
        scores_before = scoring.score_candidate(req.text, req.text, req.alpha)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring original text failed: {exc}")

    # Generate ethical rewrites using the dedicated few-shot rewrite prompt.
    # This is intentionally different from /generate and /ask which use
    # INSTRUCTION_PROMPT_TEMPLATE — keeping prompts separate per module.
    try:
        candidates_text = generation.generate_rewrite_candidates(
            req.text, req.beams, req.max_tokens
        )
    except ValueError as exc:
        # No valid candidate was produced after all retries.
        # Return a readable 422 instead of a 500 or junk output.
        logger.warning(f"/rewrite — no valid candidates: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"/rewrite — generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Rewrite generation failed: {exc}")

    try:
        scored: List[CandidateScores] = [
            scoring.score_candidate(req.text, text, req.alpha)
            for text in candidates_text
        ]
        best = max(scored, key=lambda c: c.final_score)
    except Exception as exc:
        logger.error(f"/rewrite — scoring failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Rewrite scoring failed: {exc}")

    return RewriteResponse(
        original=req.text,
        ethical_rewrite=best.text,
        scores_before=scores_before,
        scores_after=best,
    )

@app.post(
    "/analyze-chunks",
    tags=["Browser Extension"],
    summary="Score webpage text chunks — browser extension endpoint",
)
def analyze_chunks(req: ChunksInput):
    """
    Browser extension endpoint.
    Accepts webpage text as a list of chunks, scores each one,
    returns severity + optional rewrite for flagged items.

    Set auto_rewrite=False for fast initial page scan.
    Set auto_rewrite=True only when user requests corrections.
    """
    _require_models()

    if not req.chunks:
        return {"results": []}

    texts = [c.text for c in req.chunks]

    # Stage 1: batch
    tox_scores      = scoring.batch_toxicity_scores(texts)
    bias_scores     = scoring.batch_bias_scores(texts)
    manip_penalties = [scoring.get_manipulation_penalty(t) for t in texts]

    results = []

    for i, chunk in enumerate(req.chunks):
        tox_risk  = round(1.0 - tox_scores[i], 4)
        bias_risk = round(1.0 - bias_scores[i], 4)
        manip     = manip_penalties[i]

        suspicious = tox_risk > 0.12 or bias_risk > 0.12 or manip > 0

        if suspicious:
            full = scoring.score_candidate(chunk.text, chunk.text, DEFAULT_ALPHA)
            if tox_risk > 0.5 or bias_risk > 0.5 or manip > 0.3:
                severity = "HIGH"
            elif tox_risk > 0.2 or bias_risk > 0.2 or manip > 0.1 or full.ethics_score < 0.65:
                severity = "MEDIUM"
            else:
                severity = "LOW"
        else:
            severity = "LOW"

        rewritten = None
        if severity in ("HIGH", "MEDIUM") and req.auto_rewrite:
            try:
                candidates = generation.generate_rewrite_candidates(
                    chunk.text, num_candidates=2, max_tokens=60
                )
                scored = [
                    scoring.score_candidate(chunk.text, c, DEFAULT_ALPHA)
                    for c in candidates
                ]
                rewritten = max(scored, key=lambda s: s.final_score).text
            except Exception:
                rewritten = None

        results.append({
            "id": chunk.id,
            "severity": severity,
            "toxicity_risk": tox_risk,
            "bias_risk": bias_risk,
            "manipulation_penalty": manip,
            "rewritten": rewritten,
        })

    return {"results": results}