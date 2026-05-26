"""
generation.py — Model loading and text generation for EthicalGuard.

Responsibilities
----------------
1. Load the generation model (TinyLlama with distilgpt2 fallback).
2. Load all scoring models and inject them into scoring.py.
3. Expose generate_candidates() and generate_one() for use by main.py.

Why TinyLlama?
--------------
TinyLlama/TinyLlama-1.1B-Chat-v1.0 is a 1.1B parameter instruction-tuned
model that fits comfortably on a CPU or a modest GPU.  It produces far more
coherent and contextually appropriate completions than distilgpt2 (117M),
making the ethical reranking pipeline more meaningful and demonstrable.

Fallback strategy
-----------------
If TinyLlama fails to load (network issue, insufficient RAM, etc.) we
automatically fall back to distilgpt2 so the server always starts.
"""

import logging
import re
import torch
from app.config import EMPTY_OUTPUT_FALLBACK
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
# Detected once at module load so every function can reference it.
# On Colab with a T4/A100, this will be "cuda"; on a local CPU machine, "cpu".

_device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    logger.info(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available — running on CPU")


# ---------------------------------------------------------------------------
# Rewrite candidate validation
# ---------------------------------------------------------------------------

# Prompt fragments that leak into output when the model echoes the prompt
# instead of generating a rewrite.
_REWRITE_LEAK_PHRASES = [
    "input:", "output:", "assistant:", "you are an ethical ai",
    "rules:", "examples:", "now rewrite", "rewrite the following",
]


def is_valid_rewrite_candidate(text: str) -> bool:
    """
    Return True only if `text` looks like a genuine rewritten sentence.

    Rejects:
      - Empty or whitespace-only strings
      - Strings shorter than 8 characters (too short to be a sentence)
      - Strings with no alphabetic characters (pure punctuation / symbols)
      - Known junk tokens: "_", "__", "...", ".", "-", "--"
      - Prompt fragments leaking into the output (model echoed the prompt)

    Does NOT reject valid rewrites that happen to be short — the 8-char
    minimum is intentionally low to allow short but meaningful sentences.
    """
    if not text:
        return False

    stripped = text.strip()

    # Too short to be a real sentence
    if len(stripped) < 8:
        return False

    # No letters at all — pure symbols / punctuation
    if not re.search(r"[a-zA-Z]", stripped):
        return False

    # Known junk tokens
    if stripped in {"_", "__", "...", ".", "-", "--", "___"}:
        return False

    # Model echoed the prompt template instead of generating a rewrite
    lower = stripped.lower()
    if any(phrase in lower for phrase in _REWRITE_LEAK_PHRASES):
        return False

    return True

# ---------------------------------------------------------------------------
# Module-level handles (set during load_models())
# ---------------------------------------------------------------------------
gen_tokenizer = None
gen_model = None
_model_name_loaded: str = ""   # tracks which model actually loaded


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models() -> str:
    """
    Load all models required by EthicalGuard and wire them into scoring.py.

    Returns the name of the generation model that was successfully loaded.
    Raises RuntimeError if even the fallback model fails.
    """
    global gen_tokenizer, gen_model, _model_name_loaded

    # ---- 1. Generation model (phi-2 → TinyLlama → distilgpt2 fallback chain) ----
    for model_id in [GEN_MODEL_NAME, GEN_MODEL_FALLBACK, GEN_MODEL_FALLBACK2]:
        try:
            logger.info(f"Loading generation model: {model_id} on {_device} ...")
            gen_tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Use float16 + device_map on GPU for faster inference and lower VRAM usage.
            # On CPU, use float32 (float16 is not supported on most CPU backends).
            # low_cpu_mem_usage=True reduces peak RAM during loading by streaming weights.
            gen_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                device_map="auto" if _device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            # On CPU, manually move the model to the target device.
            # On GPU with device_map="auto", the model is already placed correctly.
            if _device == "cpu":
                gen_model = gen_model.to(_device)

            # Some tokenizers don't set a pad token by default.
            # We use eos_token as pad so batched generation doesn't crash.
            if gen_tokenizer.pad_token is None:
                gen_tokenizer.pad_token = gen_tokenizer.eos_token

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

    # ---- 2. Toxicity model ----
    logger.info(f"Loading toxicity model: {TOXICITY_MODEL} ...")
    try:
        reward_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        reward_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to load toxicity model ({TOXICITY_MODEL}): {exc}")

    # ---- 3. Sentiment model ----
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

    # ---- 4. Bias model ----
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

    # ---- 5. Inject all handles into scoring.py ----
    # scoring.py owns SBERT — it loads it internally inside set_scoring_models().
    scoring.set_scoring_models(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        sentiment_pipe=sentiment_pipe,
        bias_pipe=bias_pipe,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
    )

    # ---- 6. Share the SBERT model with rag.py and init vector DB ----
    # RAG reuses the same SBERT instance already loaded by scoring.py
    # so we never load it twice.
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

    The raw user prompt is wrapped in INSTRUCTION_PROMPT_TEMPLATE before
    being fed to the model.  This steers distilgpt2 (and any other causal LM)
    toward safe, respectful outputs without any fine-tuning.

    The wrapper is applied INTERNALLY here — callers always pass the original
    user prompt, and scoring always receives the original prompt too.

    Sampling parameters:
      - temperature=0.7  : moderate randomness
      - top_k=40         : sample from the 40 most likely next tokens
      - top_p=0.9        : nucleus sampling — top 90% probability mass
      - repetition_penalty=1.3 : discourages repeating phrases
      - no_repeat_ngram_size=3 : hard ban on repeating any 3-gram
    """
    # Wrap the user prompt in the instruction template.
    # For TinyLlama (chat model) we additionally apply its chat template on
    # top of the already-wrapped text so the model sees the correct format.
    wrapped = INSTRUCTION_PROMPT_TEMPLATE.format(prompt=prompt)

    if hasattr(gen_tokenizer, "apply_chat_template") and gen_tokenizer.chat_template:
        messages = [{"role": "user", "content": wrapped}]
        formatted = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = wrapped

    inputs = gen_tokenizer(formatted, return_tensors="pt")
    # Move tokenized inputs to the same device as the model.
    # On GPU this sends tensors to CUDA; on CPU this is a no-op.
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
    input_len = len(inputs["input_ids"][0])

    output = gen_model.generate(
        **inputs,
        max_length=input_len + max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=gen_tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = output[0][input_len:]
    text = gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # distilgpt2 occasionally produces zero new tokens (e.g. when the prompt
    # already fills the max_length budget, or sampling collapses to eos immediately).
    # Return a safe fallback so downstream scoring never receives empty text.
    if not text:
        logger.warning("generate_one() produced empty output — using fallback text.")
        return EMPTY_OUTPUT_FALLBACK

    return text


def generate_candidates(prompt: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate `num_candidates` non-empty completions for the prompt.

    Each call to generate_one() uses stochastic sampling, so outputs differ
    even for the same prompt — giving the reranker real choices.

    Empty outputs are filtered out and regenerated up to MAX_RETRIES times
    total to avoid an infinite loop when the model is consistently failing.
    """
    MAX_RETRIES = num_candidates * 3  # generous ceiling; stops runaway loops

    results: list[str] = []
    attempts = 0

    while len(results) < num_candidates and attempts < MAX_RETRIES:
        candidate = generate_one(prompt, max_tokens)
        attempts += 1
        # generate_one() already returns a fallback for empty outputs, but
        # we double-check here in case the fallback itself is somehow blank.
        if candidate.strip():
            results.append(candidate)
        else:
            logger.warning(f"Skipping empty candidate (attempt {attempts})")

    # If we still don't have enough after all retries, pad with the fallback.
    while len(results) < num_candidates:
        logger.warning("Padding candidates with fallback text after max retries.")
        results.append(EMPTY_OUTPUT_FALLBACK)

    return results


# ---------------------------------------------------------------------------
# Rewrite-specific generation (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------

def generate_rewrite_candidates(input_text: str, num_candidates: int, max_tokens: int) -> list[str]:
    """
    Generate rewrite candidates using the dedicated few-shot REWRITE_PROMPT_TEMPLATE.

    Only candidates that pass is_valid_rewrite_candidate() are kept.
    Invalid outputs (junk tokens, prompt echoes, pure punctuation) are logged
    and discarded — never returned to the caller.

    Raises ValueError if no valid candidate is produced after MAX_RETRIES,
    so the endpoint can return a readable HTTP error instead of junk.

    Post-processing pipeline per raw output:
      1. Strip known output prefixes ("Assistant:", "Output:", etc.)
      2. Remove surrounding quotes
      3. Remove markdown bullet characters
      4. Take the first non-empty line (model should return one sentence)
      5. Strip whitespace
      6. Validate with is_valid_rewrite_candidate()
    """
    prompt = REWRITE_PROMPT_TEMPLATE.format(input_text=input_text)

    # For chat-template models (TinyLlama), wrap the full few-shot prompt
    # as a single user message so the model sees it in the right format.
    if hasattr(gen_tokenizer, "apply_chat_template") and gen_tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    results: list[str] = []
    MAX_RETRIES = num_candidates * 5   # more retries since we're stricter now

    for attempt in range(MAX_RETRIES):
        if len(results) >= num_candidates:
            break

        inputs = gen_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
        input_len = len(inputs["input_ids"][0])

        output = gen_model.generate(
            **inputs,
            max_length=input_len + max_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=gen_tokenizer.eos_token_id,
        )

        generated_ids = output[0][input_len:]
        raw = gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.debug(f"Rewrite attempt {attempt + 1} raw output: {repr(raw)}")

        # ── Post-processing ──────────────────────────────────────────────

        cleaned = raw

        # 1. Strip known output prefixes
        for prefix in REWRITE_OUTPUT_STRIP_PREFIXES:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # 2. Remove surrounding quotes (single or double)
        cleaned = cleaned.strip('"\'')

        # 3. Remove markdown bullet characters at the start
        cleaned = re.sub(r"^[\-\*\•]\s*", "", cleaned)

        # 4. Take the first non-empty line
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = lines[0] if lines else ""

        # 5. Final whitespace strip
        cleaned = cleaned.strip()

        logger.debug(f"Rewrite attempt {attempt + 1} cleaned: {repr(cleaned)}")

        # 6. Validate
        if is_valid_rewrite_candidate(cleaned):
            results.append(cleaned)
        else:
            logger.warning(
                f"Rejected invalid rewrite candidate (attempt {attempt + 1}): {repr(cleaned)}"
            )

    if not results:
        # No valid candidate after all retries — raise so the endpoint can
        # return a readable HTTP 500 instead of junk or NaN.
        raise ValueError(
            "Could not generate a valid rewrite. "
            "The model produced only invalid outputs after multiple attempts. "
            "Please try again."
        )

    return results


# ---------------------------------------------------------------------------
# Browser-extension preparation helpers
# ---------------------------------------------------------------------------
# These functions are designed to be called by a future browser extension
# that sends webpage text to the backend for analysis and rewriting.
# They are thin wrappers that keep the extension integration surface minimal.

def analyze_webpage_text(text: str) -> list[str]:
    """
    Prepare webpage text for ethical analysis.

    Splits the text into sentence-level chunks suitable for scoring.
    Returns a list of non-empty text segments.

    Future browser extension usage:
      extension sends full page text → this function chunks it →
      each chunk is scored by scoring.score_candidate()
    """
    from app.utils import chunk_text, clean_text
    cleaned = clean_text(text)
    # Use smaller chunks for webpage text (sentences, not paragraphs)
    return chunk_text(cleaned, chunk_size=80, overlap=10)


def rewrite_webpage_chunk(chunk: str, max_tokens: int = 80) -> str:
    """
    Rewrite a single webpage text chunk into a safer version.

    Returns the best rewrite candidate (highest final_score).

    Future browser extension usage:
      user clicks "Rewrite" on a highlighted webpage section →
      extension sends the chunk here → returns the safe version
    """
    from app import scoring
    from app.config import DEFAULT_ALPHA

    candidates = generate_rewrite_candidates(chunk, num_candidates=3, max_tokens=max_tokens)
    scored = [scoring.score_candidate(chunk, c, DEFAULT_ALPHA) for c in candidates]
    best = max(scored, key=lambda s: s.final_score)
    return best.text
