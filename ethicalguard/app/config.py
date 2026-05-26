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
# Primary model: microsoft/phi-2 — a 2.7B instruction-tuned model that
# produces high-quality rewrites and analysis on CPU/GPU.
#
# Fallback chain: phi-2 → TinyLlama → distilgpt2
# Each fallback is tried automatically if the previous one fails to load.
#
# Override via environment variable:
#   set GEN_MODEL_NAME=microsoft/phi-2                          # Windows
#   export GEN_MODEL_NAME=microsoft/phi-2                       # Linux/macOS

GEN_MODEL_NAME: str = os.getenv("GEN_MODEL_NAME", "microsoft/phi-2")
GEN_MODEL_FALLBACK: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEN_MODEL_FALLBACK2: str = "distilgpt2"   # last-resort fallback for CPU-only machines

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
    "You are EthicalGuard, an AI content moderation and safety analysis system. "
    "Analyze the following content and respond in a factual, neutral, and informative way. "
    "Focus on identifying ethical issues, biases, or unsafe patterns if present.\n"
    "User: {prompt}\n"
    "Assistant:"
)

# ---------------------------------------------------------------------------
# Rewrite-specific few-shot prompt (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------
# This prompt is NOT used for /ask or /generate.
# It uses few-shot examples to teach the model to preserve meaning while
# removing manipulation, toxicity, and emotional aggression.
# The {input_text} placeholder is replaced with the user's actual text.
REWRITE_PROMPT_TEMPLATE: str = (
    "You are an ethical AI rewriting assistant.\n"
    "Your task is to rewrite harmful, manipulative, toxic, biased, or emotionally "
    "aggressive sentences into safer and more respectful versions while preserving "
    "the original intent.\n\n"
    "Rules:\n"
    "- Preserve the original meaning\n"
    "- Remove threats, manipulation, toxicity, and emotional pressure\n"
    "- Keep the rewrite concise\n"
    "- Use calm and respectful language\n"
    "- Do not add explanations\n"
    "- Return ONLY the rewritten sentence\n"
    "- Return a complete sentence with proper grammar\n"
    "- Never return symbols, punctuation-only text, or blank output\n\n"
    "Examples:\n"
    'Input: "You must listen to me or you will regret it."\n'
    'Output: "Please listen to my concerns so we can discuss this respectfully."\n\n'
    'Input: "People like them are always causing problems."\n'
    'Output: "I think there are concerns about certain behaviors that should be discussed fairly."\n\n'
    'Input: "This is the only way. Trust me blindly."\n'
    'Output: "This approach may help, but it\'s important to evaluate different options carefully."\n\n'
    "Now rewrite the following sentence safely:\n"
    'Input: "{input_text}"\n'
    "Output:"
)

# Prefixes that instruction-tuned models sometimes prepend to their output.
# These are stripped from rewrite results so only the clean sentence is returned.
REWRITE_OUTPUT_STRIP_PREFIXES: list = [
    "EthicalCoach:", "Assistant:", "Output:", "Rewrite:", "Answer:",
    "EthicalGuard:", "Safe version:", "Safer version:",
]
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
