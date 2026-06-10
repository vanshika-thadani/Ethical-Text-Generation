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
GEN_MODEL_NAME: str = os.getenv("GEN_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
GEN_MODEL_FALLBACK: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_FALLBACK2: str = "distilgpt2"

# ---------------------------------------------------------------------------
# Safety / scoring models
# ---------------------------------------------------------------------------
TOXICITY_MODEL: str = "facebook/roberta-hate-speech-dynabench-r4-target"
SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BIAS_MODEL: str = "valurank/distilroberta-bias"
SBERT_MODEL: str = os.getenv("SBERT_MODEL_PATH", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
WEIGHT_TOXICITY: float = 0.40
WEIGHT_SENTIMENT: float = 0.25
WEIGHT_BIAS: float = 0.25
WEIGHT_COHERENCE: float = 0.10

# ---------------------------------------------------------------------------
# Prompt safety gate
# ---------------------------------------------------------------------------
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
DEFAULT_MAX_TOKENS: int = 100   # for /generate, /ask, /compare
REWRITE_MAX_TOKENS: int = 60    # for /rewrite — keeps output to one sentence
DEFAULT_BEAMS: int = 5
DEFAULT_ALPHA: float = 0.7

# ---------------------------------------------------------------------------
# Instruction prompt wrapper (used by /generate and /ask)
# ---------------------------------------------------------------------------
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
# Explicit "one sentence, max 20 words" constraint stops Phi-3 from rambling.
# The constraint is stated twice (system + user) for emphasis.
REWRITE_PROMPT_TEMPLATE: str = (
    "<|system|>\n"
    "You are an ethical AI rewriting assistant. "
    "Your job is to minimally edit unsafe text, not to create new content. "
    "First decide whether the sentence actually contains toxicity, bias, manipulation, threats, coercion, or harmful stereotypes. "
    "If the sentence is already ethical, fair, and non-manipulative, return it unchanged. "
    "If rewriting is needed, preserve the original meaning and make the smallest possible edit. "
    "Do not add new facts, examples, identities, groups, topics, explanations, or assumptions. "
    "Do not expand the sentence. "
    "Output ONLY one final sentence. No quotes. No explanation.<|end|>\n"
    "<|user|>\n"
    "Review this sentence. Return it unchanged if it is already ethical. "
    "Otherwise rewrite it with minimal changes only:\n\n"
    "{input_text}<|end|>\n"
    "<|assistant|>\n"
)

# Prefixes stripped from model output in post-processing.
REWRITE_OUTPUT_STRIP_PREFIXES: list = [
    "Rewrite:", "Output:", "Assistant:", "Answer:",
    "EthicalGuard:", "Safe version:", "Safer version:", "Sentence:",
    "Rewritten sentence:", "Here is the rewritten sentence:",
]

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
CHUNK_ANALYSIS_FLAG_THRESHOLD: float = 0.6