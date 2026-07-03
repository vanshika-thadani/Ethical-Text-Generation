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
# Generation is handled by Groq API (Llama-3.3-70B).
# The model name is set in generation.py via the GROQ_MODEL env var.
# GEN_MODEL_NAME is kept here only for backwards-compat references.
# distilgpt2 is still loaded locally for fluency/perplexity scoring only.
#
# Set your API key before starting the server:
#   export GROQ_API_KEY=your_key_here   # Linux/macOS/Colab
#   set GROQ_API_KEY=your_key_here      # Windows
GEN_MODEL_NAME: str = "groq/llama-3.3-70b-versatile"   # informational only

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
# Plain text template — no Phi-3 chat tokens needed with Groq API.
# The {prompt} placeholder is filled by generation.py before calling Groq.
INSTRUCTION_PROMPT_TEMPLATE: str = (
    "Analyze the following content and respond in a factual, neutral, and informative way. "
    "Identify any ethical issues, biases, or unsafe patterns if present.\n\n"
    "{prompt}"
)

# ---------------------------------------------------------------------------
# Rewrite-specific prompt (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------
# Plain text — Groq doesn't need Phi-3 native tokens.
# The {input_text} placeholder is filled by generate_rewrite_candidates().
REWRITE_PROMPT_TEMPLATE: str = (
    "Rewrite the following sentence to be completely ethical, unbiased, and factually responsible. "
    "Check carefully for subtle bias, pseudoscience, harmful generalizations, or stereotypes "
    "about any group — even if the sentence appears neutral or scientific in tone. "
    "Always return a rewritten version. Never return the original unchanged.\n\n"
    "{input_text}"
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