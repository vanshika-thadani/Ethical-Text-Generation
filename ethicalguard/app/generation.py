"""
generation.py — Model loading and text generation for EthicalGuard.

Architecture (post-migration)
------------------------------
  Generation:   Groq API  (Llama-3.3-70B)
  Scoring:      HF Inference API (toxicity, sentiment, bias) + local CPU (SBERT, distilgpt2)

RAM on Render free tier after migration:
  distilgpt2 (fluency): ~250MB
  SBERT all-MiniLM:      ~90MB
  Total:                ~340MB  ✅ fits 512MB free tier
"""

import logging
import os
import re

from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM

from app import scoring
from app.config import (
    EMPTY_OUTPUT_FALLBACK,
    INSTRUCTION_PROMPT_TEMPLATE,
    REWRITE_PROMPT_TEMPLATE,
    REWRITE_OUTPUT_STRIP_PREFIXES,
    REWRITE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------------
GROQ_PRIMARY_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"

# distilgpt2 — loaded locally for fluency/perplexity scoring ONLY.
FLUENCY_MODEL = "distilgpt2"

# ---------------------------------------------------------------------------
# Module-level handles
# ---------------------------------------------------------------------------
_groq_client: Groq | None = None
_model_name_loaded: str = ""

# Exposed so main.py/_require_models() can verify startup completed
gen_tokenizer = None
gen_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> str:
    """
    1. Verify GROQ_API_KEY and HF_API_KEY are set.
    2. Test Groq connectivity.
    3. Load distilgpt2 locally (fluency scoring only).
    4. Wire everything into scoring.py (which initialises HF client + SBERT).

    Returns the active Groq model name.
    """
    global _groq_client, _model_name_loaded, gen_tokenizer, gen_model

    # ── 1. Groq client ───────────────────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    _groq_client = Groq(api_key=groq_key)

    active_model = None
    for model in [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]:
        try:
            _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            active_model = model
            logger.info(f"Groq client ready. Using model: {model}")
            break
        except Exception as exc:
            logger.warning(f"Groq model {model} unreachable: {exc}")

    if active_model is None:
        raise RuntimeError("Groq API unreachable. Check GROQ_API_KEY and network.")
    _model_name_loaded = active_model

    # ── 2. HF API key check (scoring.py will use it) ─────────────────────────
    hf_key = os.environ.get("HF_API_KEY")
    if not hf_key:
        raise RuntimeError(
            "HF_API_KEY not set. Get a free token at "
            "https://huggingface.co/settings/tokens"
        )
    logger.info("HF_API_KEY found — scoring models will use HF Inference API.")

    # ── 3. distilgpt2 — fluency scoring only, stays local ────────────────────
    logger.info(f"Loading fluency model: {FLUENCY_MODEL} (CPU only) ...")
    try:
        gen_tokenizer = AutoTokenizer.from_pretrained(FLUENCY_MODEL)
        gen_model = AutoModelForCausalLM.from_pretrained(FLUENCY_MODEL)
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token
        logger.info(f"Fluency model loaded: {FLUENCY_MODEL} | device: cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load fluency model ({FLUENCY_MODEL}): {exc}")

    # ── 4. Wire into scoring.py ───────────────────────────────────────────────
    # Toxicity / sentiment / bias now go through HF Inference API inside
    # scoring.py — no local model objects needed for those three.
    scoring.set_scoring_models(
        hf_api_key=hf_key,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
    )

    # ── 5. Share SBERT with rag.py and init vector DB ─────────────────────────
    from app import rag
    rag.set_rag_sbert_model(scoring._sbert_model)
    rag.init_vector_db()

    logger.info(
        f"All models ready. "
        f"Generation: Groq/{_model_name_loaded} | "
        f"Scoring: HF Inference API + local distilgpt2/SBERT"
    )
    return _model_name_loaded


# ---------------------------------------------------------------------------
# Text generation via Groq API
# ---------------------------------------------------------------------------

def generate_one(prompt: str, max_tokens: int) -> str:
    """
    Generate a single completion for the given prompt via Groq API.

    Uses a clean system message (no Phi-3 chat tokens — those were specific
    to the local model format). The INSTRUCTION_PROMPT_TEMPLATE is used as
    the user message content.

    Falls back to GROQ_FALLBACK_MODEL if the active model fails.
    """
    if _groq_client is None:
        raise RuntimeError("Groq client not initialized. Call load_models() first.")

    system_msg = (
        "You are a helpful, respectful, and ethical AI assistant. "
        "Produce safe, unbiased, non-toxic responses. "
        "Be concise and direct. Avoid harmful, manipulative, or biased language."
    )

    # Strip Phi-3 tokens from the template — Groq chat API doesn't need them.
    # The template still injects the prompt cleanly as a user turn.
    user_content = INSTRUCTION_PROMPT_TEMPLATE.format(prompt=prompt)
    # Remove any <|system|> / <|user|> / <|assistant|> / <|end|> artifacts
    user_content = re.sub(r"<\|[^|]+\|>", "", user_content).strip()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]

    for model in [_model_name_loaded, GROQ_FALLBACK_MODEL]:
        try:
            response = _groq_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            text = response.choices[0].message.content.strip()
            if text:
                return text
            logger.warning(f"generate_one(): empty response from {model}, trying fallback.")
        except Exception as exc:
            logger.warning(f"Groq generation failed ({model}): {exc}")

    logger.warning("All Groq models failed — returning fallback text.")
    return EMPTY_OUTPUT_FALLBACK


def generate_candidates(prompt: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate num_candidates non-empty completions for the prompt via Groq.

    Temperature sampling ensures diversity across candidates so the ethical
    reranker has meaningful choices to score.
    """
    MAX_RETRIES = num_candidates * 3
    results: list[str] = []
    attempts = 0

    while len(results) < num_candidates and attempts < MAX_RETRIES:
        candidate = generate_one(prompt, max_tokens)
        attempts += 1
        if candidate.strip():
            results.append(candidate)
        else:
            logger.warning(f"Skipping empty candidate (attempt {attempts})")

    while len(results) < num_candidates:
        logger.warning("Padding candidates with fallback text after max retries.")
        results.append(EMPTY_OUTPUT_FALLBACK)

    return results


# ---------------------------------------------------------------------------
# Rewrite-specific generation (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------

def _truncate_to_first_sentence(text: str) -> str:
    """
    Return the first complete sentence from text.
    Prevents LLMs from appending extra commentary after the rewrite.
    """
    match = re.match(r'^([^.!?]*[.!?])', text)
    if match:
        first = match.group(1).strip()
        if len(first.split()) >= 3:
            return first
    return text


def is_valid_rewrite_candidate(text: str, original: str = "") -> bool:
    """
    Validate a rewrite candidate. Rejects outputs that are:
      1. Empty / too short (< 5 chars or < 3 words)
      2. Pure punctuation / no letters
      3. Ends with a colon (label, not a sentence)
      4. Starts with meta-commentary ("Here is a rewrite:", etc.)
      5. Identical to the original input (no rewrite happened)
    Logs rejection reason for debugging.
    """
    if not text:
        logger.warning("Rejected rewrite: empty string")
        return False

    stripped = text.strip()

    if len(stripped) < 5:
        logger.warning(f"Rejected rewrite (too short, {len(stripped)} chars): {repr(stripped)}")
        return False

    if not re.search(r"[a-zA-Z]", stripped):
        logger.warning(f"Rejected rewrite (no letters): {repr(stripped)}")
        return False

    if len(stripped.split()) < 3:
        logger.warning(f"Rejected rewrite (< 3 words): {repr(stripped)}")
        return False

    if stripped.endswith(":"):
        logger.warning(f"Rejected rewrite (ends with colon): {repr(stripped)}")
        return False

    meta_prefix = re.compile(
        r"^("
        r"(output|input|assistant|rewritten?|safer version|example|sentence|rewrite)\s*:"
        r"|here\s+(is|are|'s)\b"
        r"|(a|the|one)\s+\w+\s+(version|way|rewrite|sentence|example)\b"
        r"|this\s+(could|would|can|is)\b"
        r"|you\s+could\s+say\b"
        r")",
        re.IGNORECASE,
    )
    if meta_prefix.match(stripped):
        logger.warning(f"Rejected rewrite (meta-prefix): {repr(stripped)}")
        return False

    if original and stripped.strip('"\'').lower() == original.strip().lower():
        logger.warning(f"Rejected rewrite (identical to original): {repr(stripped)}")
        return False

    return True


def generate_rewrite_candidates(input_text: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate rewrite candidates via Groq using REWRITE_PROMPT_TEMPLATE.

    With Llama-3.3-70B, the rewrite prompt works much better than with
    Phi-3 — the model reliably returns one clean sentence without the
    meta-commentary issues that plagued the local model.

    Post-processing pipeline per raw output:
      1. Strip known output prefixes ("Rewrite:", "Output:", etc.)
      2. Remove surrounding quotes
      3. Remove markdown bullets
      4. Take the first non-empty line
      5. Truncate to first sentence (catches any trailing commentary)
      6. Validate with is_valid_rewrite_candidate()

    Raises ValueError if no valid candidate after MAX_RETRIES.
    """
    if _groq_client is None:
        raise RuntimeError("Groq client not initialized.")

    effective_max_tokens = min(max_tokens, REWRITE_MAX_TOKENS)

    # Build a clean rewrite system message for Groq (strip Phi-3 tokens from template)
    rewrite_system = (
        "You are an ethical AI rewriting assistant. "
        "Your task is to minimally edit unsafe text. "
        "If the sentence is already ethical, return it unchanged. "
        "If rewriting is needed, preserve the original meaning with the smallest possible edit. "
        "Output ONLY one final sentence. No quotes. No explanation. No labels."
    )

    # Strip Phi-3 chat tokens from REWRITE_PROMPT_TEMPLATE — Groq doesn't need them.
    user_content = REWRITE_PROMPT_TEMPLATE.format(input_text=input_text)
    user_content = re.sub(r"<\|[^|]+\|>", "", user_content).strip()

    messages = [
        {"role": "system", "content": rewrite_system},
        {"role": "user",   "content": user_content},
    ]

    results: list[str] = []
    MAX_RETRIES = num_candidates * 3   # Groq is fast — fewer retries needed

    for attempt in range(MAX_RETRIES):
        if len(results) >= num_candidates:
            break

        try:
            response = _groq_client.chat.completions.create(
                model=_model_name_loaded,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=0.5,   # lower temp for more focused rewrites
                top_p=0.9,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"Groq rewrite attempt {attempt + 1} failed: {exc}")
            continue

        logger.debug(f"Rewrite attempt {attempt + 1} raw: {repr(raw)}")

        # ── Post-processing ──────────────────────────────────────────────
        cleaned = raw

        # 1. Strip known output prefixes
        for prefix in REWRITE_OUTPUT_STRIP_PREFIXES:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # 2. Remove surrounding quotes
        cleaned = cleaned.strip('"\'').strip()

        # 3. Remove markdown bullets
        cleaned = re.sub(r"^[\-\*\•]\s*", "", cleaned)

        # 4. First non-empty line only
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = lines[0] if lines else ""

        # 5. Truncate to first sentence
        cleaned = _truncate_to_first_sentence(cleaned)
        cleaned = cleaned.strip()

        logger.debug(f"Rewrite attempt {attempt + 1} cleaned: {repr(cleaned)}")

        if is_valid_rewrite_candidate(cleaned, original=input_text):
            results.append(cleaned)
        else:
            logger.warning(f"Rejected rewrite candidate (attempt {attempt + 1}): {repr(cleaned)}")

    if not results:
        raise ValueError(
            "Could not generate a valid rewrite after multiple attempts. "
            "Please try again."
        )

    return results


# ---------------------------------------------------------------------------
# Browser-extension preparation helpers
# ---------------------------------------------------------------------------

def analyze_webpage_text(text: str) -> list[str]:
    """
    Split webpage text into sentence-level chunks for ethical analysis.
    Used by the /analyze-chunks endpoint (browser extension).
    """
    from app.utils import chunk_text, clean_text
    cleaned = clean_text(text)
    return chunk_text(cleaned, chunk_size=80, overlap=10)


def rewrite_webpage_chunk(chunk: str, max_tokens: int = 60) -> str:
    """
    Rewrite a single webpage chunk into a safer version via Groq.
    Returns the best rewrite candidate (highest final_score).
    Used by the /analyze-chunks endpoint (browser extension).
    """
    from app import scoring
    from app.config import DEFAULT_ALPHA

    candidates = generate_rewrite_candidates(chunk, num_candidates=2, max_tokens=max_tokens)
    scored = [scoring.score_candidate(chunk, c, DEFAULT_ALPHA) for c in candidates]
    best = max(scored, key=lambda s: s.final_score)
    return best.text
