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

Fallback chain: Phi-3-mini -> TinyLlama -> distilgpt2
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
    REWRITE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    logger.info(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available — running on CPU")


# ---------------------------------------------------------------------------
# Stop token helper
# ---------------------------------------------------------------------------

def _get_model_input_device():
    """
    Resolve the correct input device for the generation model.

    With device_map="auto" (Accelerate), gen_model.device and
    next(gen_model.parameters()).device both return "meta" — a dispatch
    proxy device, not a real device. Inputs sent to "meta" cause:
      "Tensor on device cuda:0 is not on the expected device meta"

    Fix: read from hf_device_map (populated by Accelerate) to find the
    first real CUDA device. Fall back to scanning parameters, then CPU.
    """
    try:
        if hasattr(gen_model, "hf_device_map") and gen_model.hf_device_map:
            devices = list(gen_model.hf_device_map.values())
            cuda_devs = [d for d in devices if isinstance(d, int) or
                         (isinstance(d, str) and "cuda" in str(d))]
            if cuda_devs:
                d = cuda_devs[0]
                return f"cuda:{d}" if isinstance(d, int) else d
        # Fallback: first non-meta parameter
        for param in gen_model.parameters():
            if param.device.type != "meta":
                return param.device
    except Exception:
        pass
    return "cpu"
    """
    Build the list of token IDs that should stop generation.

    For Phi-3-mini: includes both EOS and <|end|> (end-of-turn token).
    Without <|end|>, the model continues past its answer into new fake turns.
    For TinyLlama / distilgpt2: only EOS is needed.
    """
    stop = [tokenizer.eos_token_id]
    try:
        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if end_id is not None and end_id != tokenizer.unk_token_id and end_id not in stop:
            stop.append(end_id)
            logger.info(f"Added <|end|> (id={end_id}) to stop tokens.")
    except Exception:
        pass
    return stop


# ---------------------------------------------------------------------------
# Sentence truncator — clips rambling output to the first complete sentence
# ---------------------------------------------------------------------------

def _truncate_to_first_sentence(text: str) -> str:
    """
    Return the first complete sentence from text.

    Phi-3 occasionally generates multiple sentences despite instructions.
    This clips everything after the first sentence-ending punctuation,
    provided the result is at least 3 words long (not a fragment).

    Examples:
      "I hope we can resolve this. Please note that..." -> "I hope we can resolve this."
      "I feel hurt." -> "I feel hurt."
      "Okay." -> unchanged (too short to be a meaningful rewrite)
    """
    match = re.match(r'^([^.!?]*[.!?])', text)
    if match:
        first = match.group(1).strip()
        if len(first.split()) >= 3:
            return first
    return text


# ---------------------------------------------------------------------------
# Rewrite candidate validation
# ---------------------------------------------------------------------------

def is_valid_rewrite_candidate(text: str, original: str = "") -> bool:
    """
    Validate a rewrite candidate.

    Rejects if ANY of these are true:
      1. Empty or whitespace-only
      2. Fewer than 5 characters
      3. Fewer than 3 words
      4. No alphabetic characters
      5. Ends with a colon (label, not a sentence)
      6. Starts with a meta-prefix (model narrating instead of rewriting)
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
_stop_tokens: list = []


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

    # ── 1. Generation model (Phi-3-mini -> TinyLlama -> distilgpt2) ─────────
    for model_id in [GEN_MODEL_NAME, GEN_MODEL_FALLBACK, GEN_MODEL_FALLBACK2]:
        try:
            logger.info(f"Loading generation model: {model_id} on {_device} ...")

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

    # ── 2. Toxicity model — force CPU to save GPU memory for generation ────
    logger.info(f"Loading toxicity model: {TOXICITY_MODEL} on CPU ...")
    try:
        reward_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            TOXICITY_MODEL
        ).to("cpu")   # explicitly on CPU — keeps VRAM free for Phi-3
        logger.info(f"Toxicity model device: cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load toxicity model ({TOXICITY_MODEL}): {exc}")

    # ── 3. Sentiment model — force CPU (device=-1 in pipeline) ──────────────
    logger.info(f"Loading sentiment model: {SENTIMENT_MODEL} on CPU ...")
    try:
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
            device=-1,   # -1 = CPU, regardless of CUDA availability
        )
        logger.info(f"Sentiment pipeline device: cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load sentiment model ({SENTIMENT_MODEL}): {exc}")

    # ── 4. Bias model — force CPU (device=-1 in pipeline) ───────────────────
    logger.info(f"Loading bias model: {BIAS_MODEL} on CPU ...")
    try:
        bias_pipe = pipeline(
            "text-classification",
            model=BIAS_MODEL,
            truncation=True,
            max_length=512,
            device=-1,   # -1 = CPU
        )
        logger.info(f"Bias pipeline device: cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load bias model ({BIAS_MODEL}): {exc}")

    # ── 5. Inject all handles into scoring.py ────────────────────────────────
    scoring.set_scoring_models(
        reward_tokenizer=reward_tokenizer,
        reward_model=reward_model,
        sentiment_pipe=sentiment_pipe,
        bias_pipe=bias_pipe,
        gen_tokenizer=gen_tokenizer,
        gen_model=gen_model,
    )

    # ── 6. Share SBERT with rag.py and init vector DB ────────────────────────
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

    INSTRUCTION_PROMPT_TEMPLATE already contains Phi-3 chat tokens.
    Do NOT apply apply_chat_template on top — that would double-format.
    """
    formatted = INSTRUCTION_PROMPT_TEMPLATE.format(prompt=prompt)

    inputs = gen_tokenizer(formatted, return_tensors="pt")
    input_device = _get_model_input_device()
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    output = gen_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=gen_tokenizer.eos_token_id,
        eos_token_id=_stop_tokens,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output[0][input_len:]
    text = gen_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not text:
        logger.warning("generate_one() produced empty output — using fallback text.")
        return EMPTY_OUTPUT_FALLBACK

    return text


def generate_candidates(prompt: str, num_candidates: int, max_tokens: int) -> list:
    """
    Generate num_candidates non-empty completions for the prompt.
    Each call uses stochastic sampling so outputs differ.
    Logs CUDA memory before/after generation for OOM diagnosis.
    """
    if _device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"generate_candidates() start — CUDA memory: "
            f"allocated={alloc:.2f}GB reserved={reserved:.2f}GB"
        )

    MAX_RETRIES = num_candidates * 3
    results = []
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

    if _device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"generate_candidates() end — CUDA memory: "
            f"allocated={alloc:.2f}GB reserved={reserved:.2f}GB"
        )

    return results


# ---------------------------------------------------------------------------
# Rewrite-specific generation (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------

def generate_rewrite_candidates(input_text: str, num_candidates: int, max_tokens: int) -> list:
    """
    Generate rewrite candidates using REWRITE_PROMPT_TEMPLATE.

    REWRITE_PROMPT_TEMPLATE already uses Phi-3 chat tokens.
    Do NOT apply apply_chat_template on top.

    Post-processing pipeline per raw output:
      1. Strip known output prefixes
      2. Strip "Output (in English):" style prefixes
      3. Remove surrounding quotes
      4. Remove markdown bullets
      5. Take first non-empty line
      5b. Truncate to first sentence (fixes Phi-3 rambling)
      6. Final strip
      7. Validate with is_valid_rewrite_candidate()

    Only candidates passing validation are kept.
    Raises ValueError if no valid candidate after MAX_RETRIES.
    """
    # Use REWRITE_MAX_TOKENS cap regardless of what caller passes in,
    # since the prompt already enforces "max 20 words" — extra tokens
    # just give the model room to ramble.
    effective_max_tokens = min(max_tokens, REWRITE_MAX_TOKENS)

    formatted = REWRITE_PROMPT_TEMPLATE.format(input_text=input_text)

    results = []
    MAX_RETRIES = num_candidates * 5

    for attempt in range(MAX_RETRIES):
        if len(results) >= num_candidates:
            break

        inputs = gen_tokenizer(formatted, return_tensors="pt")
        input_device = _get_model_input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        output = gen_model.generate(
            **inputs,
            max_new_tokens=effective_max_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=gen_tokenizer.eos_token_id,
            eos_token_id=_stop_tokens,
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

        # 2. Strip "Output (in English):" style prefixes
        cleaned = re.sub(r"(?i)^output\s*\([^)]*\)\s*:\s*", "", cleaned).strip()

        # 3. Remove surrounding quotes
        cleaned = cleaned.strip('"\'').strip()
        if cleaned.startswith('\\"') and cleaned.endswith('\\"'):
            cleaned = cleaned[2:-2].strip()

        # 4. Remove markdown bullets
        cleaned = re.sub(r"^[\-\*\•]\s*", "", cleaned)

        # 5. Take first non-empty line
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = lines[0] if lines else ""

        # 5b. Truncate to first sentence — fixes Phi-3 rambling past
        #     the answer even when max_new_tokens is respected.
        #     "I hope we resolve this. Please note that complex..." -> "I hope we resolve this."
        cleaned = _truncate_to_first_sentence(cleaned)

        # 6. Final strip
        cleaned = cleaned.strip()

        logger.debug(f"Rewrite attempt {attempt + 1} cleaned: {repr(cleaned)}")

        # 7. Validate
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

def analyze_webpage_text(text: str) -> list:
    """
    Prepare webpage text for ethical analysis.
    Splits into sentence-level chunks suitable for scoring.
    """
    from app.utils import chunk_text, clean_text
    cleaned = clean_text(text)
    return chunk_text(cleaned, chunk_size=80, overlap=10)


def rewrite_webpage_chunk(chunk: str, max_tokens: int = 60) -> str:
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