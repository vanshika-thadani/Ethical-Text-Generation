"""
config.py — Centralized configuration for EthicalGuard.

All environment variables and tunable constants live here.
Import this module anywhere you need a setting — never hardcode values.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Generation model
# ---------------------------------------------------------------------------
# Primary model: microsoft/Phi-3-mini-4k-instruct
#   - 3.8B parameters, instruction-tuned (SFT + DPO + RLHF)
#   - Fits on Colab T4 GPU (~7.5 GB VRAM in float16)
#   - Uses native chat tokens: <|system|> <|user|> <|assistant|> <|end|>
#   - Requires trust_remote_code=True
#
# Fallback chain: Phi-3-mini → TinyLlama → distilgpt2
# Each fallback is tried automatically if the previous one fails to load.
#
# On Colab: set env var to Drive cache path so it loads in ~90s instead of
# re-downloading every session:
#   os.environ["GEN_MODEL_NAME"] = "/content/drive/MyDrive/ethicalguard_models/phi3-mini-instruct"

GEN_MODEL_NAME: str = os.getenv("GEN_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
GEN_MODEL_FALLBACK: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_FALLBACK2: str = "distilgpt2"   # last-resort fallback for CPU-only machines

# ---------------------------------------------------------------------------
# Safety / scoring models (HuggingFace model IDs)
# ---------------------------------------------------------------------------
TOXICITY_MODEL: str = "facebook/roberta-hate-speech-dynabench-r4-target"
SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BIAS_MODEL: str = "valurank/distilroberta-bias"

# SBERT_MODEL_PATH env var lets you point at a Drive-cached copy on Colab:
#   os.environ["SBERT_MODEL_PATH"] = "/content/drive/MyDrive/ethicalguard_models/all-MiniLM-L6-v2"
SBERT_MODEL: str = os.getenv("SBERT_MODEL_PATH", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
# Ethics composite = w_tox * toxicity + w_sent * sentiment + w_bias * bias + w_coh * coherence
WEIGHT_TOXICITY: float = 0.40
WEIGHT_SENTIMENT: float = 0.25
WEIGHT_BIAS: float = 0.25
WEIGHT_COHERENCE: float = 0.10

# ---------------------------------------------------------------------------
# Prompt safety gate
# ---------------------------------------------------------------------------
# If the prompt's own toxicity risk exceeds this threshold, block generation.
PROMPT_BLOCK_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Manipulation penalty
# ---------------------------------------------------------------------------
MANIPULATION_TRIGGERS: list = [
    "always",
    "everyone",
    "must",
    "no choice",
    "guaranteed",
    "secret",
    "don't tell anyone",
    "you will regret",
    "only way",
    "trust me blindly",
]
MANIPULATION_PENALTY_PER_WORD: float = 0.1
MANIPULATION_PENALTY_MAX: float = 0.5

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS: int = 100   # increased from 50 — Phi-3 needs more room
DEFAULT_BEAMS: int = 5
DEFAULT_ALPHA: float = 0.7

# ---------------------------------------------------------------------------
# Instruction prompt wrapper (used by /generate and /ask)
# ---------------------------------------------------------------------------
# Uses Phi-3 native chat tokens: <|system|>, <|user|>, <|assistant|>, <|end|>
#
# IMPORTANT:
#   - Do NOT apply apply_chat_template() on top of this in generation.py.
#     The tokens are already embedded in the string.
#   - Scoring always compares against the ORIGINAL user prompt, not this
#     wrapper, so coherence scores remain meaningful.
#   - TinyLlama / distilgpt2 will see the raw tokens as text — not ideal,
#     but they will still produce coherent output. The fallback is intentional.

INSTRUCTION_PROMPT_TEMPLATE: str = (
    "<|system|>\n"
    "You are EthicalGuard, an AI safety and content analysis assistant. "
    "Respond in a factual, neutral, and informative way. "
    "Identify ethical issues, biases, or unsafe patterns if present.<|end|>\n"
    "<|user|>\n"
    "{prompt}<|end|>\n"
    "<|assistant|>\n"
)

# ---------------------------------------------------------------------------
# Rewrite-specific prompt (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------
# Uses Phi-3 native chat tokens — do NOT apply apply_chat_template() on top.
#
# Design decisions:
#   - System message is short and precise: one job, one output format.
#   - "Output ONLY the rewritten sentence" is the most important instruction.
#     Phi-3 respects this reliably; phi-2 did not.
#   - No few-shot examples needed — Phi-3 follows instructions without them.
#     (Few-shot examples actually caused phi-2 to continue the list pattern
#     instead of rewriting. Phi-3 doesn't have this problem.)
#   - <|end|> after <|user|> closes the user turn cleanly.
#   - <|assistant|> with no content prompts the model to start generating.

REWRITE_PROMPT_TEMPLATE: str = (
    "<|system|>\n"
    "You are an ethical AI rewriting assistant. "
    "Rewrite the given sentence to remove toxic, manipulative, threatening, or harmful language "
    "while keeping the speaker's core feeling or concern intact. "
    "Output ONLY the rewritten sentence — no explanation, no label, no quotes, nothing else.<|end|>\n"
    "<|user|>\n"
    "Rewrite this sentence ethically:\n\n"
    "{input_text}<|end|>\n"
    "<|assistant|>\n"
)

# Prefixes that models sometimes prepend to their output.
# These are stripped in generation.py post-processing.
# Phi-3 rarely produces these, but they're kept for fallback models.
REWRITE_OUTPUT_STRIP_PREFIXES: list = [
    "Rewrite:", "Output:", "Assistant:", "Answer:",
    "EthicalGuard:", "Safe version:", "Safer version:", "Sentence:",
    "Rewritten sentence:", "Here is the rewritten sentence:",
]

# Substituted when the model produces zero tokens (rare with Phi-3).
EMPTY_OUTPUT_FALLBACK: str = "I recommend expressing your feelings honestly and respectfully."

# ---------------------------------------------------------------------------
# RAG / Vector DB settings
# ---------------------------------------------------------------------------
VECTORDB_PATH: str = os.getenv("VECTORDB_PATH", "vectordb")
RAG_COLLECTION_NAME: str = "ethicalguard_docs"
RAG_TOP_K: int = 3
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 80
UPLOAD_DIR: str = "uploads"

# Chunks with ethics_score below this threshold are flagged as unsafe
# in the /analyze-document endpoint.
CHUNK_ANALYSIS_FLAG_THRESHOLD: float = 0.6