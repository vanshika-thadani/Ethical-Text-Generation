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
GEN_MODEL_NAME: str = "groq/openai/gpt-oss-120b"   # informational only

# ---------------------------------------------------------------------------
# Safety / scoring models
# ---------------------------------------------------------------------------
TOXICITY_MODEL: str = "s-nlp/roberta_toxicity_classifier"
SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BIAS_MODEL: str = "valurank/distilroberta-bias"
SBERT_MODEL: str = os.getenv("SBERT_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")

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
DEFAULT_MAX_TOKENS: int = 300   # for /generate, /ask, /compare — enough for a full answer
REWRITE_MAX_TOKENS: int = 200   # for /rewrite — raised from 80 to handle multi-sentence inputs
DEFAULT_BEAMS: int = 5
DEFAULT_ALPHA: float = 0.7

# ---------------------------------------------------------------------------
# Instruction prompt wrapper (used by /generate and /ask)
# ---------------------------------------------------------------------------
# Rules for the model:
#  - Answer in plain prose, NO markdown tables, NO bullet lists unless asked
#  - Be specific and cite the text where relevant
#  - Stay concise: 3-5 sentences max
INSTRUCTION_PROMPT_TEMPLATE: str = (
    "You are EthicalGuard, an AI content safety analyst. "
    "Answer the following question clearly and concisely in plain prose (no markdown tables, "
    "no bullet points unless the question explicitly asks for a list). "
    "Be specific: quote or paraphrase the relevant parts of the text when supporting your answer. "
    "Keep your answer to 3-5 sentences.\n\n"
    "{prompt}"
)

# ---------------------------------------------------------------------------
# Rewrite-specific prompt (used ONLY by /rewrite endpoint)
# ---------------------------------------------------------------------------
# Rules for the model:
#  - Output ONE sentence only — the rewritten version
#  - No labels, no explanations, no quotes around the output
#  - Preserve the speaker's core meaning; only remove harmful/biased language
REWRITE_PROMPT_TEMPLATE: str = (
    "You are a text safety editor. The sentence below has been flagged as potentially toxic, "
    "biased, or manipulative. Rewrite it to remove the harmful element while preserving the "
    "speaker's core meaning as closely as possible. "
    "If the sentence is genuinely harmless after careful reading, return it unchanged. "
    "Output ONLY the single rewritten sentence — no explanation, no label, no quotes.\n\n"
    "Original: {input_text}\n"
    "Rewritten:"
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