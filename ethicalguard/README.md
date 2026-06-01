# 🛡️ EthicalGuard – RAG-Powered Ethical Content Analysis & Safe Rewriting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)

> A training-free AI safety platform combining ethical LLM reranking, RAG-powered document analysis, unsafe content detection, and safe rewriting — all through a clean REST API and React frontend. Designed as the backend for a future browser extension.

---

## What EthicalGuard Does

EthicalGuard is **not a chatbot**. It is an ethical content moderation and analysis platform that:

- **Analyzes documents** for toxic, biased, and manipulative content chunk by chunk
- **Rewrites unsafe text** into safer, respectful versions using few-shot prompted LLMs
- **Answers questions** about uploaded documents using RAG (Retrieval-Augmented Generation)
- **Compares** raw LLM output against ethically reranked output side by side
- **Scores** every piece of text across 5 safety dimensions without any fine-tuning

---

## Architecture

```
User uploads document (txt / pdf)
    │
    ▼
┌─────────────────────────────────┐
│  ingestion.py                   │
│  extract → clean → chunk        │  ~400 words/chunk, 80-word overlap
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

User asks a question / requests analysis
    │
    ▼  embed with SBERT → nearest-neighbour search → top-k chunks
    │
    ▼  LLM generates candidates → ethical reranking → safest answer returned
```

### Ethical Reranking Pipeline

```
User Prompt
    │
    ▼  Prompt safety gate (block if toxicity risk > 0.7)
    │
    ▼  Instruction wrapper → LLM generates N candidates
    │
    ▼  Score each candidate:
    │     Toxicity  (40%) → RoBERTa hate-speech (facebook/roberta-hate-speech-dynabench-r4-target)
    │     Sentiment (25%) → Twitter RoBERTa     (cardiffnlp/twitter-roberta-base-sentiment-latest)
    │     Bias      (25%) → DistilRoBERTa       (valurank/distilroberta-bias)
    │     Coherence (10%) → SBERT cosine similarity
    │     Fluency        → perplexity from generation model
    │     Manipulation   → trigger phrase penalty (capped at 0.5)
    │
    └── Final = α × ethics + (1-α) × fluency − manipulation_penalty
              → best candidate returned
```

---

## Key Features

| Feature | Details |
|---------|---------|
| **Ethical reranking** | Generates N candidates, scores all, returns safest |
| **RAG document QA** | Upload → chunk → embed → retrieve → answer |
| **Document analysis** | Per-chunk severity: HIGH / MEDIUM / LOW |
| **Safe rewriting** | Few-shot prompted rewriting preserving original meaning |
| **Baseline comparison** | Raw LLM vs safety-ranked output side by side |
| **Prompt safety gate** | Blocks harmful prompts before generation |
| **NaN safety** | `safe_float()` guards on every score — no JSON crashes |
| **GPU support** | float16 + `device_map="auto"` on CUDA, float32 on CPU |
| **Browser-extension ready** | `analyze_webpage_text()` and `rewrite_webpage_chunk()` stubs |

---

## Severity Classification

Severity is based on **raw risk signals**, not the composite ethics score:

| Severity | Condition |
|----------|-----------|
| **HIGH** | `toxicity_risk > 0.7` OR `manipulation_penalty > 0.4` OR `bias_risk > 0.6` |
| **MEDIUM** | `toxicity_risk > 0.4` OR `bias_risk > 0.3` |
| **LOW** | otherwise (safe) |

Where `toxicity_risk = 1 - toxicity_score` and `bias_risk = 1 - bias_score`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 |
| Generation (primary) | `microsoft/phi-2` |
| Generation (fallback 1) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Generation (fallback 2) | `distilgpt2` |
| Toxicity | `facebook/roberta-hate-speech-dynabench-r4-target` |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Bias | `valurank/distilroberta-bias` |
| Embeddings + coherence | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (faiss-cpu) |
| PDF extraction | pdfplumber + PyPDF2 fallback |
| Validation | Pydantic v2 |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS v4 |

---

## Project Structure

```
ethicalguard/
├── app/
│   ├── config.py       ← all constants, model names, prompt templates, env vars
│   ├── models.py       ← Pydantic request/response schemas
│   ├── utils.py        ← text cleaning and chunking helpers
│   ├── ingestion.py    ← document extraction pipeline (txt/pdf)
│   ├── rag.py          ← FAISS vector store + retrieval
│   ├── scoring.py      ← all safety metrics, safe_float guards, NaN protection
│   ├── generation.py   ← model loading, generation, rewrite pipeline, browser-ext stubs
│   └── main.py         ← FastAPI app, all 8 endpoints
├── data/
│   └── eval_prompts.csv   ← 25 benchmark prompts across 5 categories
├── uploads/               ← uploaded files (gitignored)
├── vectordb/              ← FAISS index (gitignored, rebuilt from uploads)
├── results/               ← benchmark output (gitignored)
├── evaluate.py            ← automated benchmark runner
├── requirements.txt
└── README.md

ethicalguard-ui/
├── src/
│   ├── components/        ← 10 React components
│   ├── services/api.ts    ← Axios instance, VITE_API_URL support
│   └── types/api.ts       ← TypeScript interfaces for all endpoints
├── .env.local             ← gitignored — set VITE_API_URL here for Colab/ngrok
└── vite.config.ts
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Health check |
| `GET` | `/rag-status` | Vector DB stats and document list |
| `POST` | `/upload` | Ingest a .txt or .pdf document |
| `POST` | `/ask` | RAG question answering over uploaded documents |
| `POST` | `/analyze-document` | Per-chunk ethical analysis with severity labels |
| `POST` | `/rewrite` | Rewrite toxic/manipulative text into safer version |
| `POST` | `/generate` | Ethical reranked text generation (prompt blocked if unsafe) |
| `POST` | `/compare` | Baseline vs safety-ranked output comparison |

---

## How to Run Locally

```bash
# 1. Clone
git clone https://github.com/vanshika-thadani/Ethical-Text-Generation.git
cd Ethical-Text-Generation/ethicalguard

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start backend
python -m uvicorn app.main:app --reload
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (Swagger UI)

# 5. Start frontend (separate terminal)
cd ../ethicalguard-ui
npm install
npm run dev
# → http://localhost:5173
```

**Switch generation model (optional):**
```bash
# Windows
set GEN_MODEL_NAME=microsoft/phi-2

# Linux / macOS
export GEN_MODEL_NAME=microsoft/phi-2
```

---

## Running on Google Colab (GPU)

```python
# In Colab — start the backend
import subprocess
subprocess.Popen([
    "python", "-m", "uvicorn", "app.main:app",
    "--host", "0.0.0.0", "--port", "8000"
])

# Get a public URL via cloudflared (no account needed)
import subprocess, time, re
proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stderr=subprocess.PIPE
)
time.sleep(5)
for _ in range(20):
    line = proc.stderr.readline().decode()
    if "trycloudflare.com" in line:
        url = re.search(r'https://\S+\.trycloudflare\.com', line)
        if url:
            print("Set this in ethicalguard-ui/.env.local:")
            print(f"VITE_API_URL={url.group()}")
            break
```

Then in `ethicalguard-ui/.env.local`:
```
VITE_API_URL=https://xxxx.trycloudflare.com
```

Restart the frontend dev server after updating `.env.local`.

---

## Running the Benchmark

```bash
# Terminal 1 — backend
python -m uvicorn app.main:app --reload

# Terminal 2 — benchmark
python evaluate.py
```

Results saved to:
- `results/evaluation_results.csv` — per-prompt scores for baseline and safety-ranked
- `results/summary.json` — averages by category

---

## Rewrite Prompt Design

The `/rewrite` endpoint uses a dedicated few-shot prompt (`REWRITE_PROMPT_TEMPLATE`) that is **separate** from `/ask` and `/generate`. It teaches the model to:

- **Keep** the speaker's underlying emotion, concern, or complaint
- **Remove** personal attacks, threats, manipulation, and insults
- **Replace** blame with first-person expression: `"you ruined this"` → `"this situation did not go well"`

Candidate validation rejects outputs that are:
- Fewer than 5 characters or 3 words
- Pure punctuation or symbols
- Meta-commentary (`"Here is a safer version:"`, `"The rewritten sentence:"`)
- Identical to the original input

If no valid candidate is produced after all retries, the endpoint returns HTTP 422 with a readable message instead of junk output.

---

## Future Work

- [ ] Browser extension (Chrome/Firefox) for real-time webpage analysis
- [ ] Streaming generation support (SSE)
- [ ] Image moderation module
- [ ] Factuality scoring dimension
- [ ] Docker + docker-compose deployment
- [ ] User feedback loop for online preference learning

---

## References

- Christiano et al. (2017). Deep RL from Human Preferences. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Ouyang et al. (2022). InstructGPT. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bender et al. (2021). On the Dangers of Stochastic Parrots. ACM FAccT.
- Johnson et al. (2019). Billion-scale similarity search with FAISS. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
