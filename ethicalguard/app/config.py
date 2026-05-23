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
# Default is distilgpt2 for lightweight local CPU development and testing.
# distilgpt2 is a 117M parameter model that loads quickly and runs on CPU
# without requiring significant RAM or a GPU.
#
# To switch to TinyLlama on a more capable machine (GPU / Colab / Kaggle),
# set the environment variable before starting the server:
#   export GEN_MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Linux/macOS
#   set GEN_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0         # Windows
#
# Falls back to distilgpt2 if the primary model fails to load.

GEN_MODEL_NAME: str = os.getenv("GEN_MODEL_NAME", "distilgpt2")
GEN_MODEL_FALLBACK: str = "distilgpt2"

# ---------------------------------------------------------------------------
# Safety / scoring models (HuggingFace model IDs)
# ---------------------------------------------------------------------------
TOXICITY_MODEL: str = "facebook/roberta-hate-speech-dynabench-r4-target"
SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BIAS_MODEL: str = "valurank/distilroberta-bias"
SBERT_MODEL: str = "all-MiniLM-L6-v2"

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
# Each trigger word found in the text adds MANIPULATION_PENALTY_PER_WORD.
# Total penalty is capped at MANIPULATION_PENALTY_MAX.
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
DEFAULT_MAX_TOKENS: int = 50
DEFAULT_BEAMS: int = 5
DEFAULT_ALPHA: float = 0.7

# ---------------------------------------------------------------------------
# Instruction prompt wrapper
# ---------------------------------------------------------------------------
# Wraps the raw user prompt before passing it to the generation model.
# This steers even lightweight models like distilgpt2 toward safe, respectful
# outputs without any fine-tuning.
#
# IMPORTANT: this wrapped prompt is used ONLY for generation.
# Scoring (coherence, toxicity, etc.) always compares against the original
# user prompt so results remain meaningful and unbiased by the wrapper.
INSTRUCTION_PROMPT_TEMPLATE: str = (
    "You are EthicalGuard, a safe and respectful communication coach. "
    "Respond in a calm, non-manipulative, emotionally healthy way. "
    "If the user asks for guilt, revenge, manipulation, or harmful persuasion, "
    "redirect them toward honest and respectful communication.\n"
    "User: {prompt}\n"
    "Assistant:"
)
# distilgpt2 occasionally generates zero new tokens (e.g. when the prompt
# fills the max_length budget or sampling collapses to eos immediately).
# This neutral, safe sentence is substituted whenever that happens so the
# scoring pipeline never receives an empty string.
EMPTY_OUTPUT_FALLBACK: str = "I recommend expressing your feelings honestly and respectfully."

# ---------------------------------------------------------------------------
# RAG / Vector DB settings
# ---------------------------------------------------------------------------
# VECTORDB_PATH      : where FAISS index files are persisted between restarts.
# RAG_COLLECTION_NAME: logical name (used as a label, not a DB collection).
# RAG_TOP_K          : default number of chunks to retrieve per question.
# CHUNK_SIZE         : target words per chunk during document ingestion.
# CHUNK_OVERLAP      : words shared between consecutive chunks (preserves
#                      context at boundaries so no sentence is cut in half).
# UPLOAD_DIR         : where uploaded files are saved before ingestion.

VECTORDB_PATH: str = os.getenv("VECTORDB_PATH", "vectordb")
RAG_COLLECTION_NAME: str = "ethicalguard_docs"
RAG_TOP_K: int = 3
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 80
UPLOAD_DIR: str = "uploads"

# Chunks with ethics_score below this threshold are flagged as unsafe
# in the /analyze-document endpoint.
CHUNK_ANALYSIS_FLAG_THRESHOLD: float = 0.6
