"""
scoring.py — All safety and quality scoring logic for EthicalGuard.

Design principle: score_candidate() is the single entry point.
It computes every metric once and returns a CandidateScores object.
No other module should call individual scoring helpers directly —
always go through score_candidate() to avoid redundant computation.

Metrics explained
-----------------
toxicity_score  : How hate-speech-free the text is (1 = safe).
                  Uses a RoBERTa model fine-tuned on the DynaBench hate-speech
                  dataset.  We take the probability of the "non-hate" class.

sentiment_score : Penalises strongly negative tone.
                  Uses a Twitter-trained RoBERTa sentiment model.
                  Positive → ~1.0, Neutral → 0.8, Negative → 0.5–0.75.

bias_score      : How free the text is from stereotyped / prejudiced language.
                  Uses a DistilRoBERTa model trained on the MBIC bias corpus.
                  1.0 = unbiased, 0.0 = strongly biased.

coherence_score : Semantic similarity between prompt and response.
                  Computed with SBERT cosine similarity.
                  High score means the response actually addresses the prompt.

ethics_score    : Weighted composite of the four scores above.
                  Weights are defined in config.py.

fluency_score   : Perplexity-based naturalness score.
                  Lower perplexity → more natural text → higher score.
                  We reuse the generation model already in memory — no extra
                  download needed.

manipulation_penalty : Deducted from the final score when the text contains
                       known manipulation trigger phrases (see config.py).

final_score     : alpha * ethics_score + (1 - alpha) * fluency_score
                  - manipulation_penalty
                  Higher is better.  alpha controls the ethics/fluency trade-off.
"""

from __future__ import annotations

import math
import torch
from sentence_transformers import SentenceTransformer, util
from app.config import (
    WEIGHT_TOXICITY, WEIGHT_SENTIMENT, WEIGHT_BIAS, WEIGHT_COHERENCE,
    MANIPULATION_TRIGGERS, MANIPULATION_PENALTY_PER_WORD, MANIPULATION_PENALTY_MAX,
    SBERT_MODEL, EMPTY_OUTPUT_FALLBACK,
)
from app.models import CandidateScores


# ---------------------------------------------------------------------------
# Module-level model handles (populated by generation.py after model load)
# ---------------------------------------------------------------------------
# We store references here so scoring.py never imports generation.py
# (which would create a circular dependency).  generation.py calls
# scoring.set_gen_model() once after its own models are ready.

_reward_tokenizer = None
_reward_model = None
_sentiment_pipeline = None
_bias_pipeline = None
_sbert_model = None  # type: SentenceTransformer | None
_gen_tokenizer = None
_gen_model = None

# Resolved at startup: index of the safe (non-toxic) class in the toxicity model's output.
# Never assume a fixed index — different model checkpoints use different label orders.
_safe_label_index: int = 0


# Keywords used to identify which labels are "safe" vs "unsafe" in the toxicity model.
# Safe labels: nothate, non-hate, not hate, normal, neutral
# Unsafe labels: hate, toxic, offensive, abusive
_SAFE_LABEL_KEYWORDS = {"nothate", "non-hate", "not hate", "normal", "neutral"}
_UNSAFE_LABEL_KEYWORDS = {"hate", "toxic", "offensive", "abusive"}


import logging as _logging
_logger = _logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NaN / Inf safety helper
# ---------------------------------------------------------------------------

def safe_float(value: object, default: float = 0.0, label: str = "") -> float:
    """
    Convert a raw score to a JSON-safe float.

    Returns `default` (0.0) if the value is:
      - None
      - NaN  (can occur when a model returns degenerate logits)
      - Inf  (can occur from log(0) or division by zero in perplexity)

    Logs a warning whenever a replacement is made so the root cause can be
    investigated without crashing the API.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        _logger.warning(f"safe_float: could not convert {label!r} value {value!r} → using {default}")
        return default

    if math.isnan(v) or math.isinf(v):
        _logger.warning(f"safe_float: {label!r} is {'NaN' if math.isnan(v) else 'Inf'} → replacing with {default}")
        return default

    return round(v, 4)


def _resolve_safe_label_index(model) -> int:
    """
    Inspect model.config.id2label to find which output index corresponds
    to the safe / non-toxic class.

    Logic:
      - If a label contains any SAFE keyword  → treat as safe index.
      - If a label contains any UNSAFE keyword → treat as unsafe index.
      - If neither matches, fall back to index 0 and log a warning.

    Logs the full id2label map once at startup so label order is always
    visible in the server logs for debugging.
    """
    id2label = model.config.id2label  # e.g. {0: 'nothate', 1: 'hate'}
    _logger.info(f"Toxicity model id2label: {id2label}")

    for idx, label in id2label.items():
        label_lower = label.lower().replace("_", " ").replace("-", " ")
        if any(kw in label_lower for kw in _SAFE_LABEL_KEYWORDS):
            _logger.info(f"Safe label identified: index={idx}, label='{label}'")
            return int(idx)

    # No safe label found by keyword — fall back and warn
    _logger.warning(
        f"Could not identify a safe label by keyword in id2label={id2label}. "
        f"Falling back to index 0. Scores may be inaccurate."
    )
    return 0


def set_scoring_models(
    reward_tokenizer,
    reward_model,
    sentiment_pipe,
    bias_pipe,
    gen_tokenizer,
    gen_model,
):
    """
    Called once at startup (from generation.py) to inject all model handles
    into this module.  Avoids re-loading models and prevents circular imports.

    Also resolves the safe-label index for the toxicity model so
    _toxicity_score() never hard-codes an assumption about label order.
    """
    global _reward_tokenizer, _reward_model
    global _sentiment_pipeline, _bias_pipeline
    global _gen_tokenizer, _gen_model
    global _sbert_model, _safe_label_index

    _reward_tokenizer = reward_tokenizer
    _reward_model = reward_model
    _sentiment_pipeline = sentiment_pipe
    _bias_pipeline = bias_pipe
    _gen_tokenizer = gen_tokenizer
    _gen_model = gen_model

    # Resolve which output index is the "safe" class — do this once at startup.
    _safe_label_index = _resolve_safe_label_index(reward_model)

    # SBERT is lightweight — load it here so scoring.py owns it fully.
    _sbert_model = SentenceTransformer(SBERT_MODEL)


# ---------------------------------------------------------------------------
# Individual metric helpers (private — use score_candidate() externally)
# ---------------------------------------------------------------------------

def _toxicity_score(text: str) -> float:
    """
    Returns a value in [0, 1] where 1.0 means the text is completely safe
    and 0.0 means it is maximally toxic / hateful.

    How it works:
      The RoBERTa hate-speech model outputs one logit per class.
      After softmax, we extract the probability of the SAFE class using
      _safe_label_index, which is resolved at startup by inspecting
      model.config.id2label — never assumed to be a fixed index.

      For facebook/roberta-hate-speech-dynabench-r4-target:
        id2label = {0: 'nothate', 1: 'hate'}
        → _safe_label_index = 0
        → toxicity_score = softmax(logits)[0]  (probability of 'nothate')
    """
    inputs = _reward_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = _reward_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # Use the dynamically resolved safe-class index
    safe_prob = probs[_safe_label_index].item()
    return round(safe_prob, 4)


def _sentiment_score(text: str) -> float:
    """
    Returns a value in [0, 1] reflecting tone quality.

    Scoring logic:
      - Positive sentiment → 0.8 + 0.2 * confidence  (max 1.0)
      - Neutral sentiment  → 0.8
      - Negative sentiment → 1.0 - 0.5 * confidence  (min ~0.5)

    We penalise negative tone because ethically generated text should
    avoid unnecessarily distressing or hostile language.
    """
    result = _sentiment_pipeline(text[:512])[0]
    label = result["label"].lower()
    conf = result["score"]

    if "negative" in label:
        return round(1.0 - conf * 0.5, 4)
    elif "positive" in label:
        return round(min(1.0, 0.8 + conf * 0.2), 4)
    else:
        return 0.8


def _bias_score(text: str) -> float:
    """
    Returns a value in [0, 1] where 1.0 = completely unbiased.

    The DistilRoBERTa bias model outputs a label ("biased" / "not biased")
    and a confidence score.  We invert the confidence when the label is
    "biased" so that higher always means safer.
    """
    result = _bias_pipeline(text[:512])[0]
    if result["label"].lower() == "biased":
        return round(1.0 - result["score"], 4)
    return round(result["score"], 4)


def _coherence_score(prompt: str, response: str) -> float:
    """
    Returns cosine similarity between SBERT embeddings of prompt and response.

    Range: [0, 1] after clamping (raw cosine can be slightly negative).
    High score = the response is semantically on-topic.

    Guards:
      - Empty prompt or response → return 0.0 (no meaningful similarity)
      - NaN cosine result        → return 0.0 (degenerate embedding)
    """
    if not prompt or not prompt.strip() or not response or not response.strip():
        return 0.0

    prompt_emb = _sbert_model.encode(prompt, convert_to_tensor=True)
    resp_emb = _sbert_model.encode(response, convert_to_tensor=True)
    sim = float(util.cos_sim(prompt_emb, resp_emb).item())

    # Clamp to [0, 1] and guard against NaN from degenerate embeddings
    if math.isnan(sim) or math.isinf(sim):
        _logger.warning(f"_coherence_score: cosine similarity is NaN/Inf → returning 0.0")
        return 0.0

    return round(max(0.0, min(sim, 1.0)), 4)


def _fluency_score(text: str) -> float:
    """
    Perplexity-based fluency score in [0, 1].

    Perplexity measures how "surprised" the language model is by the text.
    A well-formed English sentence has low perplexity; garbled or repetitive
    text has high perplexity.

    We reuse the generation model (already in memory) as the evaluator:
      1. Tokenise the text.
      2. Run a forward pass with the text as both input AND labels.
         PyTorch computes cross-entropy loss automatically.
      3. Convert loss → perplexity = exp(loss).
      4. Map to [0, 1]: score = 1 / (1 + log(perplexity)).
         log() keeps the score meaningful across a wide perplexity range.

    Empty text is returned as 0.0 immediately — tokenising an empty string
    produces a zero-element tensor that crashes the reshape inside the model.
    """
    # Guard: empty or whitespace-only text cannot be tokenised safely.
    if not text or not text.strip():
        return 0.0

    inputs = _gen_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    # Move inputs to the same device as the generation model.
    # This is required when the model is on GPU — CPU tensors cannot be
    # passed to a CUDA model.
    model_device = next(_gen_model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _gen_model(**inputs, labels=inputs["input_ids"])

    loss = outputs.loss.item()

    # Guard: NaN or Inf loss means the model produced degenerate output
    if math.isnan(loss) or math.isinf(loss):
        _logger.warning(f"_fluency_score: loss is {'NaN' if math.isnan(loss) else 'Inf'} → returning 0.0")
        return 0.0

    # Guard: unusually high loss (> 20) indicates near-random text.
    # exp(20) ≈ 485M perplexity — effectively meaningless, return low score.
    if loss > 20:
        _logger.warning(f"_fluency_score: unusually high loss={loss:.2f} → returning 0.1")
        return 0.1

    perplexity = math.exp(min(loss, 100.0))   # cap before exp to avoid overflow
    score = 1.0 / (1.0 + math.log(max(perplexity, 1.0)))
    return safe_float(score, default=0.0, label="fluency_score")


def _manipulation_penalty(text: str) -> float:
    """
    Scans text for manipulation trigger phrases and returns a penalty value.

    Each trigger phrase found adds MANIPULATION_PENALTY_PER_WORD (0.1).
    The total is capped at MANIPULATION_PENALTY_MAX (0.5).

    Trigger phrases are defined in config.py and include coercive language
    like "no choice", "trust me blindly", "you will regret", etc.
    These patterns are common in misinformation, scam content, and
    emotionally manipulative writing.
    """
    text_lower = text.lower()
    penalty = sum(
        MANIPULATION_PENALTY_PER_WORD
        for trigger in MANIPULATION_TRIGGERS
        if trigger in text_lower
    )
    return round(min(penalty, MANIPULATION_PENALTY_MAX), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(prompt: str, text: str, alpha: float) -> CandidateScores:
    """
    Compute all safety and quality metrics for a single candidate text.

    This is the ONLY function other modules should call.
    All metrics are computed exactly once and bundled into a CandidateScores
    object, which is then reused everywhere (no redundant inference calls).

    Parameters
    ----------
    prompt : str
        The original user prompt (needed for coherence scoring).
    text : str
        The candidate completion to evaluate.  If empty, replaced with
        EMPTY_OUTPUT_FALLBACK before scoring so no downstream function
        ever receives a blank string.
    alpha : float
        Weight for ethics vs fluency in the final score.
        alpha=1.0 → pure ethics, alpha=0.0 → pure fluency.

    Returns
    -------
    CandidateScores
        A fully populated score object ready to be serialised and returned
        in API responses or written to evaluation CSVs.
    """
    # Guard: replace empty candidates before any metric touches the text.
    # This is a last-resort safety net; generate_one() and generate_candidates()
    # should already prevent empty strings from reaching here.
    if not text or not text.strip():
        _logger.warning("score_candidate() received empty text — substituting fallback.")
        text = EMPTY_OUTPUT_FALLBACK

    tox   = safe_float(_toxicity_score(text),          label="toxicity_score")
    sent  = safe_float(_sentiment_score(text),          label="sentiment_score")
    bias  = safe_float(_bias_score(text),               label="bias_score")
    coh   = safe_float(_coherence_score(prompt, text),  label="coherence_score")
    flu   = safe_float(_fluency_score(text),            label="fluency_score")
    manip = safe_float(_manipulation_penalty(text),     label="manipulation_penalty")

    # Ethics composite — weighted average of the four safety dimensions
    ethics = safe_float(
        WEIGHT_TOXICITY * tox
        + WEIGHT_SENTIMENT * sent
        + WEIGHT_BIAS * bias
        + WEIGHT_COHERENCE * coh,
        label="ethics_score",
    )

    # Final score balances ethics and fluency, then subtracts manipulation
    final = safe_float(
        alpha * ethics + (1 - alpha) * flu - manip,
        label="final_score",
    )

    return CandidateScores(
        text=text,
        toxicity_score=tox,
        sentiment_score=sent,
        bias_score=bias,
        coherence_score=coh,
        ethics_score=ethics,
        fluency_score=flu,
        manipulation_penalty=manip,
        final_score=final,
    )


def score_prompt_risk(prompt: str) -> float:
    """
    Assess the toxicity risk of the prompt itself (before any generation).

    Returns a risk value in [0, 1] where:
      0.0 = completely safe prompt
      1.0 = maximally toxic / harmful prompt

    This is the INVERSE of _toxicity_score because we want "risk" to be
    high when the text is dangerous (toxicity_score is high when safe).
    """
    return round(1.0 - _toxicity_score(prompt), 4)
