# 🛡️ Ethical Text Generation
### Multi-Dimensional Inference-Time Reranking

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org)

> A training-free safety layer for LLMs. Generates N candidates and selects the most ethical one using 4-dimensional scoring — no fine-tuning, no human annotation required.

---

## How It Works

```
Prompt → DistilGPT-2 → N Candidates
                              ↓
         Toxicity (40%) + Sentiment (25%) + Bias (25%) + Coherence (10%)
                              ↓
         Final Score = α × ethics + (1-α) × fluency − manipulation_penalty
                              ↓
                        Best Candidate
```

---

## Setup

```bash
git clone https://github.com/vanshika-thadani/Ethical-Text-Generation.git
cd ethical-text-gen
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

---

## API

**POST** `/generate`

```json
{
  "text": "People who disagree with me are",
  "max_tokens": 50,
  "beams": 5,
  "alpha": 0.7
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `text` | required | Input prompt |
| `max_tokens` | 50 | Length of generation |
| `beams` | 5 | Number of candidates |
| `alpha` | 0.7 | Ethics vs fluency weight (0–1) |

---

## Models Used

| Task | Model |
|------|-------|
| Generation | `distilgpt2` |
| Toxicity | `facebook/roberta-hate-speech-dynabench-r4-target` |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Bias | `valurank/distilroberta-bias` |
| Coherence | `sentence-transformers/all-MiniLM-L6-v2` |

---

## References

- Christiano et al. (2017). Deep RL from Human Preferences. NeurIPS. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Ouyang et al. (2022). InstructGPT. NeurIPS. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Bender et al. (2021). On the Dangers of Stochastic Parrots. ACM FAccT.