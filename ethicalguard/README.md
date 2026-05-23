# 🛡️ EthicalGuard – Safety-Aware LLM + RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)

> A training-free AI safety backend combining ethical LLM reranking with RAG-powered document analysis. Upload documents, ask questions, detect unsafe content, and rewrite it — all through a clean REST API.

---

## Overview

EthicalGuard v3 extends the original ethical reranking system with a full **RAG (Retrieval-Augmented Generation)** pipeline. Users can upload documents, ask questions grounded in those documents, detect unethical or manipulative sections, and generate safer rewrites — without any model fine-tuning.

Designed as the backend for a future browser extension that will analyse and rewrite webpage content in real time.

---

## Key Features

| Feature | Endpoint |
|---------|----------|
| Ethical reranked text generation | `POST /generate` |
| Baseline vs ethical comparison | `POST /compare` |
| Document upload & ingestion | `POST /upload` |
| RAG question answering | `POST /ask` |
| Ethical document analysis | `POST /analyze-document` |
| Safe text rewriting | `POST /rewrite` |
| Vector DB status | `GET /rag-status` |

---

## RAG Architecture

```
User uploads document (txt / pdf)
    │
    ▼
┌─────────────────────────────────┐
│  ingestion.py                   │
│  extract text → clean → chunk   │  ~400 words/chunk, 80-word overlap
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  SBERT (all-MiniLM-L6-v2)       │  384-dim embeddings
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  FAISS IndexFlatL2              │  persisted to vectordb/
└─────────────────────────────────┘

User asks a question
    │
    ▼  embed question with SBERT
    │
    ▼  nearest-neighbour search in FAISS → top-k chunks
    │
    ▼  build context prompt → LLM generates answer
    │
    ▼  ethical reranking → safest answer returned
```

---

## Upload Flow

```
POST /upload  (multipart/form-data, file=<txt or pdf>)
    │
    ├── save to uploads/
    ├── extract text (pdfplumber → PyPDF2 fallback)
    ├── clean + chunk (~400 words, 80-word overlap)
    ├── embed with SBERT
    └── upsert into FAISS (re-upload is safe — no duplicates)
```

---

## Retrieval Flow

```
POST /ask  { "question": "...", "top_k": 3 }
    │
    ├── embed question with SBERT
    ├── FAISS L2 search → top-k chunks
    ├── build RAG prompt: context + question
    ├── generate N candidates with LLM
    └── ethical reranking → best answer + scores returned
```

---

## Ethical Analysis Flow

```
POST /analyze-document  { "document_name": "report.pdf" }
    │
    ├── load all chunks for document from FAISS metadata
    ├── score each chunk: toxicity, bias, manipulation, ethics
    └── flag chunks where ethics_score < 0.6
```

---

## Ethical Reranking (core pipeline)

```
User Prompt
    │
    ▼  Prompt safety gate (block if toxicity risk > 0.7)
    │
    ▼  Instruction wrapper → LLM generates N candidates
    │
    ▼  Score each candidate:
    │     Toxicity  (40%) → RoBERTa hate-speech
    │     Sentiment (25%) → Twitter RoBERTa
    │     Bias      (25%) → DistilRoBERTa
    │     Coherence (10%) → SBERT cosine similarity
    │     Fluency        → perplexity from gen model
    │     Manipulation   → trigger phrase penalty
    │
    └── Final = α × ethics + (1-α) × fluency − penalty
              → best candidate returned
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 |
| Generation | distilgpt2 (default) / TinyLlama-1.1B-Chat |
| Toxicity | facebook/roberta-hate-speech-dynabench-r4-target |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment-latest |
| Bias | valurank/distilroberta-bias |
| Embeddings + coherence | sentence-transformers/all-MiniLM-L6-v2 |
| Vector search | FAISS (faiss-cpu) |
| PDF extraction | pdfplumber + PyPDF2 fallback |
| Validation | Pydantic v2 |

---

## Project Structure

```
ethicalguard/
├── app/
│   ├── config.py       ← all constants, model names, env vars
│   ├── models.py       ← Pydantic request/response schemas
│   ├── utils.py        ← text cleaning and chunking helpers
│   ├── ingestion.py    ← document extraction pipeline (txt/pdf)
│   ├── rag.py          ← FAISS vector store + retrieval
│   ├── scoring.py      ← all safety metrics
│   ├── generation.py   ← model loading + text generation
│   └── main.py         ← FastAPI app, all endpoints
├── data/
│   └── eval_prompts.csv
├── uploads/            ← uploaded files (gitignored)
├── vectordb/           ← FAISS index (gitignored, rebuilt from uploads)
├── results/            ← benchmark output (gitignored)
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## How to Run Locally

```bash
git clone https://github.com/vanshika-thadani/Ethical-Text-Generation.git
cd Ethical-Text-Generation/ethicalguard

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

API: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

**Switch to TinyLlama (GPU/Colab):**
```bash
set GEN_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0   # Windows
export GEN_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 # Linux
```

---

## API Examples

### Upload a document
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_document.pdf"
```
```json
{ "status": "success", "document_name": "my_document.pdf", "chunks_added": 12 }
```

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risks mentioned?", "top_k": 3}'
```
```json
{
  "question": "What are the main risks mentioned?",
  "retrieved_chunks": [
    { "text": "...", "document": "my_document.pdf", "chunk_index": 4, "distance": 0.312 }
  ],
  "answer": "The main risks include...",
  "ethical_scores": { "toxicity_score": 0.99, "final_score": 0.74 }
}
```

### Analyze a document for unsafe content
```bash
curl -X POST http://localhost:8000/analyze-document \
  -H "Content-Type: application/json" \
  -d '{"document_name": "my_document.pdf"}'
```
```json
{
  "document_name": "my_document.pdf",
  "total_chunks": 12,
  "flagged_chunks": 2,
  "unsafe_chunks": [
    { "chunk": "...", "toxicity_score": 0.41, "bias_score": 0.38, "flagged": true }
  ]
}
```

### Rewrite unsafe text
```bash
curl -X POST http://localhost:8000/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text": "You must do this or you will regret it."}'
```
```json
{
  "original": "You must do this or you will regret it.",
  "ethical_rewrite": "I encourage you to consider this carefully...",
  "scores_before": { "final_score": 0.20 },
  "scores_after":  { "final_score": 0.57 }
}
```

---

## Running the Benchmark

```bash
# Terminal 1
python -m uvicorn app.main:app --reload

# Terminal 2
python evaluate.py
```

---

## Future: Browser Extension

The API is designed to support a browser extension that will:
- Send selected webpage text to `/analyze-document` for ethical analysis
- Highlight toxic/biased/manipulative sections
- Offer one-click rewriting via `/rewrite`
- Show safety scores inline on the page

All endpoints accept plain JSON — no browser-specific changes needed.

---

## References

- Christiano et al. (2017). Deep RL from Human Preferences. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Ouyang et al. (2022). InstructGPT. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bender et al. (2021). On the Dangers of Stochastic Parrots. ACM FAccT.
- Johnson et al. (2019). Billion-scale similarity search with FAISS. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
