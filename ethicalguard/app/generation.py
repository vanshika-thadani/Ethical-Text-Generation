"""
generation.py — Model loading and text generation for EthicalGuard.

Responsibilities
----------------
1. Load the generation model (Phi-3-mini-4k-instruct with fallback chain).
2. Load all scoring models and inject them into scoring.py.
3. Expose generate_candidates() and generate_one() for use by main.py.

Primary model: microsoft/Phi-3-mini-4k-instruct
------------------------------------------------
Phi-3-mini is a 3.8B instruction-tuned model (SFT + DPO + RLHF) that fits
comfortably on a Colab T4 GPU (~7.5 GB VRAM in float16), leaving enough
headroom for the scoring models (RoBERTa, SBERT, DistilRoBERTa).

Key differences from phi-2 / TinyLlama:
  - Uses native chat tokens: <|system|>, <|user|>, <|assistant|>, <|end|>
  - Requires trust_remote_code=True to load
  - Use max_new_tokens (not max_length) — prompt tokens are long due to
    the chat template; max_length would leave almost no room for generation
  - Must include <|end|> in eos_token_id to prevent rambling past the answer

Fallback chain: Phi-3-mini → TinyLlama → distilgpt2
Each fallback is tried automatically if the previous one fails to load.
"""

import logging
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)

from app import scoring
from app.config import (
    GEN_MODEL_NAME,
    GEN_MODEL_FALLBACK,
    GEN_MODEL_FALLBACK2,
    TOXICITY_MODEL,
    SENTIMENT_MODEL,
    BIAS_MODEL,
    EMPTY_OUTPUT_FALLBACK,
    INSTRUCTION_PROMPT_TEMPLATE,
    REWRITE_PROMPT_TEMPLATE,
    REWRITE_OUTPUT_STRIP_PREFIXES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection — use GPU if available, fall back to CPU
# ---------------------------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    logger.info(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available — running on CPU")


# ---------------------------------------------------------------------------
# Stop token helper
# ---------------------------------------------------------------------------

def _build_stop_tokens(tokenizer) -> list[int]:
    """
    Build the list of token IDs that should stop generation.

    For Phi-3-mini: includes both the standard EOS token and <|end|>
    (the end-of-turn token). Without <|end|>, the model continues generating
    beyond its answer into repeated instructions or new fake turns.

    For TinyLlama / distilgpt2: only EOS is needed — they don't use <|end|>.
    """
    stop = [tokenizer.eos_token_id]

    # Try to add <|end|> — only present in Phi-3 tokenizers
    try:
        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        # convert_tokens_to_ids returns unk_token_id when the token doesn't exist
        if end_id is not None and end_id != tokenizer.unk_token_id and end_id not in stop:
            stop.append(end_id)
            logger.info(f"Added <|end|> (id={end_id}) to stop tokens.")
    except Exception:
        pass  # silently skip for models that don't have this token

    return stop


# ---------------------------------------------------------------------------
# Rewrite candidate validation
# ---------------------------------------------------------------------------

def is_valid_rewrite_candidate(text: str, original: str = "") -> bool:
    """
    Validate a rewrite candidate with relaxed rules so concise valid rewrites
    are not rejected.

    Rejects if ANY of these are true:
      1. Empty or whitespace-only
      2. Fewer than 5 characters
      3. Fewer than 3 words
      4. No alphabetic characters at all
      5. Ends with a colon (label/header, not a sentence)
      6. Starts with a meta-prefix pattern (model narrating instead of rewriting)
      7. Identical to the original input (no rewrite happened)
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

    words = stripped.split()
    if len(words) < 3:
        logger.warning(f"Rejected rewrite (only {len(words)} words): {repr(stripped)}")
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


# ---------------------------------------------------------------------------
# Module-level handles (set during load_models())
# ---------------------------------------------------------------------------
gen_tokenizer = None
gen_model = None
_model_name_loaded: str = ""
_stop_tokens: list[int] = []   # populated after tokenizer loads


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> str:
    """
    Load all models required by EthicalGuard and wire them into scoring.py.

    Returns the name of the generation model that was successfully loaded.
    Raises RuntimeError if even the fallback model fails.
    """
    global gen_tokenizer, gen_model, _model_name_loaded, _stop_tokens

    # ── 1. Generation model (Phi-3-mini → TinyLlama → distilgpt2) ──────────
    for model_id in [GEN_MODEL_NAME, GEN_MODEL_FALLBACK, GEN_MODEL_FALLBACK2]:
        try:
            logger.info(f"Loading generation model: {model_id} on {_device} ...")

            # trust_remote_code=True is required for Phi-3 models.
            # It is harmless for TinyLlama and distilgpt2.
            gen_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            gen_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                device_map="auto" if _device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if _device == "cpu":
                gen_model = gen_model.to(_device)

            if gen_tokenizer.pad_token is None:
                gen_tokenizer.pad_token = gen_tokenizer.eos_token

            # Build stop tokens once after the tokenizer is ready
            _stop_tokens = _build_stop_tokens(gen_tokenizer)
            logger.info(f"Stop tokens: {_stop_tokens}")

            _model_name_loaded = model_id
            logger.info(
                f"Generation model loaded: {model_id} | "
                f"device: {next(gen_model.parameters()).device}"
            )
            break

        except Exception as exc:
            logger.warning(f"Failed to load {model_id}: {exc}")
            if model_id == GEN_MODEL_FALLBACK2:
                raise RuntimeError(
                    f"Could not load any generation model. "
                    f"Tried: {GEN_MODEL_NAME}, {GEN_MODEL_FALLBACK}, {GEN_MODEL_FALLBACK2}. "
                    f"Last error: {exc}"
                )

    # ── 2. Toxicity model ───────────────────────────────────────────────────
    logger.info(f"Loading toxicity model: {TOXICITY_MODEL} ...")
    try:
        reward_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        reward_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to load toxicity model ({TOXICITY_MODEL}): {exc}")

    # ── 3. Sentiment model ──────────────────────────────────────────────────
    logger.info(f"Loading sentiment model: {SENTIMENT_MODEL} ...")
    try:
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load sentiment model ({SENTIMENT_MODEL}): {exc}")

    # ── 4. Bias model ───────────────────────────────────────────────────────
    logger.info(f"Loading bias model: {BIAS_MODEL} ...")
    try:
        bias_pipe = pipeline(
            "text-classification",
            model=BIAS_MODEL,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load bias model ({BIAS_MODEL}): {exc}")

    # ── 5. Inject all handles into scoring.py ───────────────────────────────
    scoring.set_scoring_models(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        sentiment_pipe=sentiment_pipe,
        bias_pipe=bias_pipe,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
    )

    # ── 6. Share SBERT with rag.py and init vector DB ───────────────────────
    from app import rag
    rag.set_rag_sbert_model(scoring._sbert_model)
    rag.init_vector_db()

    logger.info("All models loaded successfully.")
    return _model_name_loaded


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def generate_one(prompt: str, max_tokens: int) -> str:
    """
    Generate a single completion for the given prompt.

    Uses INSTRUCTION_PROMPT_TEMPLATE which already contains Phi-3 chat tokens.
    Do NOT apply apply_chat_template on top — that would double-format the prompt.

    Key changes vs phi-2 version:
      - max_new_tokens instead of max_length (prompt is long due to chat template)
      - eos_token_id includes <|end|> to stop Phi-3 from rambling
    """
    formatted = INSTRUCTION_PROMPT_TEMPLATE.format(prompt=prompt)

    inputs = gen_tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

    output = gen_model.generate(
        **inputs,
        max_new_tokens=max_tokens,          # ← was max_length; max_new_tokens is correct for chat models
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=gen_tokenizer.eos_token_id,
        eos_token_id=_stop_tokens,          # ← stops at <|end|> for Phi-3
    )

    # Decode only newly generated tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    text = gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not text:
        logger.warning("generate_one() produced empty output — using fallback text.")
        return EMPTY_OUTPUT_FALLBACK

    return text


def generate_candidates(prompt: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate `num_candidates` non-empty completions for the prompt.

    Each call uses stochastic sampling so outputs differ — giving the
    reranker real choices.
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

def generate_rewrite_candidates(input_text: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate rewrite candidates using the dedicated REWRITE_PROMPT_TEMPLATE.

    REWRITE_PROMPT_TEMPLATE already uses Phi-3 chat tokens — do NOT apply
    apply_chat_template on top.

    Only candidates that pass is_valid_rewrite_candidate() are kept.
    Raises ValueError if no valid candidate is produced after MAX_RETRIES.
    """
    formatted = REWRITE_PROMPT_TEMPLATE.format(input_text=input_text)

    results: list[str] = []
    MAX_RETRIES = num_candidates * 5

    for attempt in range(MAX_RETRIES):
        if len(results) >= num_candidates:
            break

        inputs = gen_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

        output = gen_model.generate(
            **inputs,
            max_new_tokens=max_tokens,      # ← max_new_tokens, not max_length
            do_sample=True,
            temperature=0.6,                # slightly lower temp for more focused rewrites
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=gen_tokenizer.eos_token_id,
            eos_token_id=_stop_tokens,      # ← stops at <|end|> for Phi-3
        )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        raw = gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.debug(f"Rewrite attempt {attempt + 1} raw output: {repr(raw)}")

        # ── Post-processing ──────────────────────────────────────────────

        cleaned = raw

        # 1. Strip known output prefixes
        for prefix in REWRITE_OUTPUT_STRIP_PREFIXES:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # 1b. Strip "Output (in English):" style prefixes
        cleaned = re.sub(r"(?i)^output\s*\([^)]*\)\s*:\s*", "", cleaned).strip()

        # 2. Remove surrounding quotes
        cleaned = cleaned.strip('"\'').strip()
        if cleaned.startswith('\\"') and cleaned.endswith('\\"'):
            cleaned = cleaned[2:-2].strip()

        # 3. Remove markdown bullets
        cleaned = re.sub(r"^[\-\*\•]\s*", "", cleaned)

        # 4. Take the first non-empty line only
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = lines[0] if lines else ""

        # 5. Final strip
        cleaned = cleaned.strip()

        logger.debug(f"Rewrite attempt {attempt + 1} cleaned: {repr(cleaned)}")

        # 6. Validate
        if is_valid_rewrite_candidate(cleaned, original=input_text):
            results.append(cleaned)
        else:
            logger.warning(
                f"Rejected invalid rewrite candidate (attempt {attempt + 1}): {repr(cleaned)}"
            )

    if not results:
        raise ValueError(
            "Could not generate a valid rewrite. "
            "The model produced only invalid outputs after multiple attempts. "
            "Please try again."
        )

    return results


# ---------------------------------------------------------------------------
# Browser-extension preparation helpers
# ---------------------------------------------------------------------------

def analyze_webpage_text(text: str) -> list[str]:
    """
    Prepare webpage text for ethical analysis.

    Splits text into sentence-level chunks suitable for scoring.
    Returns a list of non-empty text segments.
    """
    from app.utils import chunk_text, clean_text
    cleaned = clean_text(text)
    return chunk_text(cleaned, chunk_size=80, overlap=10)


def rewrite_webpage_chunk(chunk: str, max_tokens: int = 80) -> str:
    """
    Rewrite a single webpage text chunk into a safer version.

    Returns the best rewrite candidate (highest final_score).
    """
    from app import scoring
    from app.config import DEFAULT_ALPHA

    candidates = generate_rewrite_candidates(chunk, num_candidates=3, max_tokens=max_tokens)
    scored = [scoring.score_candidate(chunk, c, DEFAULT_ALPHA) for c in candidates]
    best = max(scored, key=lambda s: s.final_score)
    return best.text