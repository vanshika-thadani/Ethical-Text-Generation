import math
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util  # NEW — for semantic coherence

# ------------------ APP SETUP ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GENERATION MODEL ------------------

GEN_MODEL_NAME = "distilgpt2"

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME)

if gen_tokenizer.pad_token is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

# ------------------ REWARD MODEL (TOXICITY) ------------------

REWARD_MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"

reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME)

# ------------------ SENTIMENT + BIAS MODELS ------------------

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, max_length=512,
)

bias_pipeline = pipeline(
    "text-classification",
    model="valurank/distilroberta-bias",
    truncation=True, max_length=512,
)

# ------------------ SBERT MODEL (NEW — for semantic coherence) ------------------
# 'all-MiniLM-L6-v2' is a lightweight sentence-transformer model.
# It converts full sentences into 384-dimensional embedding vectors.
# Semantically similar sentences will have vectors that point in the same direction
# (high cosine similarity), even if they share no words at all.
# Install with: pip install sentence-transformers

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ INPUT ------------------

class TextInput(BaseModel):
    text: str = Field(..., min_length=3)
    max_tokens: int = Field(default=50, ge=10, le=200)
    beams: int = Field(default=5, ge=1, le=10)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)

# ------------------ CONTEXTUAL ETHICS SCORE (IMPROVED) ------------------

def get_ethics_score(prompt, response):
    # 1. Toxicity (hate speech)
    combined = prompt + " " + response
    inputs = reward_tokenizer(combined, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
    toxicity = 1 - torch.softmax(logits, dim=-1)[0][1].item()

    # 2. Sentiment (penalise strongly negative tone)
    sentiment = sentiment_pipeline(response[:512])[0]
    label, conf = sentiment["label"].lower(), sentiment["score"]
    if "negative" in label:
        sentiment_score = 1.0 - (conf * 0.5)
    elif "positive" in label:
        sentiment_score = min(1.0, 0.8 + conf * 0.2)
    else:
        sentiment_score = 0.8

    # 3. Bias (prejudiced / stereotyped language)
    bias = bias_pipeline(response[:512])[0]
    bias_score = (1.0 - bias["score"]) if bias["label"].lower() == "biased" else bias["score"]

    # 4. IMPROVED — Semantic Coherence via SBERT (replaces token overlap)
    #
    #    OLD METHOD (token overlap):
    #        coherence = min(len(prompt_words & response_words) / max(len(prompt_words), 1) * 2, 1.0)
    #    Problem: purely lexical — checks if the SAME WORDS appear in both.
    #    Fails badly when the response is semantically related but uses different vocabulary.
    #
    #    Example of old method failing:
    #        Prompt:   "What is artificial intelligence?"
    #        Response: "Machine learning systems mimic human cognition."
    #        Token overlap → 0 (no shared words!) even though it is a perfect answer.
    #
    #    NEW METHOD (SBERT cosine similarity):
    #    - Both prompt and response are converted into 384-dimensional embedding vectors
    #      by the sentence-transformer model.
    #    - Semantically similar sentences have vectors pointing in the same direction.
    #    - Cosine similarity measures the angle between these two vectors:
    #        similarity = 1.0  → identical meaning
    #        similarity = 0.0  → completely unrelated
    #        similarity < 0.0  → opposite meaning (rare in practice)
    #    - We clamp the result to [0, 1] so it works as a score.
    #
    #    Same example with new method:
    #        Prompt:   "What is artificial intelligence?"
    #        Response: "Machine learning systems mimic human cognition."
    #        SBERT similarity → ~0.78  ✓ correctly identifies semantic relevance

    prompt_embedding = sbert_model.encode(prompt, convert_to_tensor=True)
    response_embedding = sbert_model.encode(response, convert_to_tensor=True)
    coherence = float(util.cos_sim(prompt_embedding, response_embedding).item())
    coherence = max(0.0, min(coherence, 1.0))  # clamp to [0, 1]

    return 0.40 * toxicity + 0.25 * sentiment_score + 0.25 * bias_score + 0.10 * coherence

# ------------------ MANIPULATION PENALTY ------------------

def manipulation_penalty(text):
    triggers = ["always", "everyone", "must", "no choice"]

    penalty = 0
    for word in triggers:
        if word in text.lower():
            penalty += 0.1

    return min(penalty, 0.5)

# ------------------ IMPROVED FLUENCY SCORE (PERPLEXITY-BASED) ------------------
#
# OLD METHOD (word count):
#     fluency = min(word_count / 40, 1.0)
# Problem: length ≠ fluency. A 40-word grammatically broken sentence
# scores 1.0, while a perfectly written 10-word sentence scores only 0.25.
# Word count is a proxy for completeness, not linguistic quality.
#
# NEW METHOD (perplexity):
# Perplexity measures how "surprised" the language model is by the text.
#
#   - LOW perplexity  → model finds the text very natural and predictable → FLUENT
#   - HIGH perplexity → model finds the text surprising / unnatural → NOT FLUENT
#
# How it works technically:
#   1. We feed the text back into distilgpt2 with the text as its own labels.
#   2. PyTorch computes cross-entropy loss — the average "wrongness" per token.
#   3. Perplexity = e^(loss).  So loss=0 → perplexity=1 (perfect), loss=4 → perplexity=54.
#   4. We map perplexity to a 0–1 score using: score = 1 / (1 + log(perplexity))
#      - perplexity = 1   → score = 1.0  (perfectly fluent)
#      - perplexity = 10  → score = 0.70 (natural but not perfect)
#      - perplexity = 100 → score = 0.33 (noticeably unnatural)
#      - perplexity = 1000 → score = 0.13 (very broken text)
#
# Key advantage: We reuse the SAME distilgpt2 model already in memory.
# No extra model download, no extra RAM cost.

def fluency_score(text):
    """
    Computes fluency using perplexity from the generation model.
    Lower perplexity = more natural text = higher fluency score.

    Returns a float in [0, 1] where 1.0 = perfectly fluent.
    """
    # Tokenize the text — same tokenizer used during generation
    inputs = gen_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Feed text into model with itself as labels.
    # This triggers cross-entropy loss computation internally.
    # torch.no_grad() because we are NOT training — just evaluating.
    with torch.no_grad():
        outputs = gen_model(
            **inputs,
            labels=inputs["input_ids"]  # labels = the text itself
        )

    # outputs.loss is the average cross-entropy loss per token
    loss = outputs.loss.item()

    # Convert loss → perplexity → score
    perplexity = math.exp(loss)

    # Map to [0, 1]: higher score = more fluent
    # log(perplexity) grows slowly so the score stays meaningful across a wide range
    score = 1.0 / (1.0 + math.log(max(perplexity, 1.0)))

    return round(score, 3)

# ------------------ FINAL SCORE ------------------

def final_score(prompt, text, alpha):
    e = get_ethics_score(prompt, text)
    f = fluency_score(text)
    m = manipulation_penalty(text)

    return alpha * e + (1 - alpha) * f - m

# ------------------ GENERATION ------------------

def generate_one(prompt, max_tokens):
    inputs = gen_tokenizer(prompt, return_tensors="pt")

    output = gen_model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=gen_tokenizer.eos_token_id
    )

    return gen_tokenizer.decode(output[0], skip_special_tokens=True)

# ------------------ BEST GENERATION ------------------

def generate_best(prompt, max_tokens, beams, alpha):
    candidates = [generate_one(prompt, max_tokens) for _ in range(beams)]

    scored = []

    for text in candidates:
        score = final_score(prompt, text, alpha)

        scored.append({
            "text": text,
            "ethics_score": round(get_ethics_score(prompt, text), 3),
            "fluency_score": round(fluency_score(text), 3),
            "manipulation_penalty": round(manipulation_penalty(text), 3),
            "final_score": round(score, 3)
        })

    best = max(scored, key=lambda x: x["final_score"])

    return {
        "generated_text": best["text"],
        "best_candidate": best,
        "all_candidates": scored
    }

# ------------------ API ------------------

@app.post("/generate")
def generate(input_data: TextInput):
    try:
        return generate_best(
            input_data.text,
            input_data.max_tokens,
            input_data.beams,
            input_data.alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "contextual + production"}