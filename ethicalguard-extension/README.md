# EthicalGuard Browser Extension

Highlights toxic, biased, and manipulative text on any webpage with a red wavy underline and shows a safe AI-generated replacement on hover.

## Setup

### 1. Generate icons (one-time)
```bash
cd ethicalguard-extension
python3 create_icons.py
```

### 2. Load in Chrome
1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `ethicalguard-extension/` folder

### 3. Make sure the backend is running
```bash
cd ethicalguard
source venv/bin/activate
GROQ_API_KEY=... HF_API_KEY=... python -m uvicorn app.main:app --reload
```

The extension defaults to `http://127.0.0.1:8000`. You can change the backend URL in the popup.

## How it works

1. On page load, the content script collects all visible text nodes (sentences ≥ 4 words)
2. Sends them in batches to `POST /analyze-chunks` on the backend
3. The backend scores each chunk for toxicity, bias, and manipulation
4. HIGH and MEDIUM chunks also get a safe rewrite generated automatically
5. Flagged text is wrapped with a red wavy underline (`text-decoration: underline wavy red`)
6. Hovering a flagged sentence shows a dark tooltip with:
   - Risk level (HIGH / MEDIUM)
   - Reason (toxic / biased / manipulative language)
   - ✦ Suggested replacement (green text)
   - "Copy replacement" button

## Popup features
- Enable/disable toggle
- High / Medium / Safe counts for the current page  
- Re-scan button (rescans after page content changes)
- Clear highlights button
- Configurable backend URL (for Colab/ngrok deployments)
