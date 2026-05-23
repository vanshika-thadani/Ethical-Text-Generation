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
    TOXICITY_MODEL,
    SENTIMENT_MODEL,
    BIAS_MODEL,
    EMPTY_OUTPUT_FALLBACK,
    INSTRUCTION_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

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

    # ---- 1. Generation model (TinyLlama → distilgpt2 fallback) ----
    for model_id in [GEN_MODEL_NAME, GEN_MODEL_FALLBACK]:
        try:
            logger.info(f"Loading generation model: {model_id} ...")
            gen_tokenizer = AutoTokenizer.from_pretrained(model_id)
            gen_model = AutoModelForCausalLM.from_pretrained(model_id)

            # Some tokenizers (e.g. TinyLlama) don't set a pad token by default.
            # We use eos_token as pad so batched generation doesn't crash.
            if gen_tokenizer.pad_token is None:
                gen_tokenizer.pad_token = gen_tokenizer.eos_token

            _model_name_loaded = model_id
            logger.info(f"Generation model loaded: {model_id}")
            break
        except Exception as exc:
            logger.warning(f"Failed to load {model_id}: {exc}")
            if model_id == GEN_MODEL_FALLBACK:
                raise RuntimeError(
                    f"Could not load any generation model. "
                    f"Tried: {GEN_MODEL_NAME}, {GEN_MODEL_FALLBACK}. "
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
