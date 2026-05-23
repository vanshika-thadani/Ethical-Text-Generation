"""
evaluate.py — Automated benchmark evaluation for EthicalGuard.

PURPOSE
-------
This script is NOT for end users.
It is a research / demonstration tool that:
  1. Reads prompts from data/eval_prompts.csv
  2. Calls the local FastAPI /compare endpoint for each prompt
  3. Saves per-prompt results to results/evaluation_results.csv
  4. Computes aggregate averages and saves them to results/summary.json

This lets you quantitatively demonstrate that ethical reranking improves
output safety compared to a raw (baseline) LLM generation.

HOW TO RUN
----------
1. Start the FastAPI server in a separate terminal:
       uvicorn app.main:app --reload

2. Then run this script:
       python evaluate.py

OUTPUTS
-------
results/evaluation_results.csv  — one row per prompt
results/summary.json            — overall and per-category averages

METRICS EXPLAINED
-----------------
toxicity_score      : 1.0 = safe, 0.0 = toxic  (higher is better)
bias_score          : 1.0 = unbiased            (higher is better)
final_score         : combined ethics + fluency  (higher is better)
toxicity_reduction  : baseline_tox - ranked_tox  (positive = improvement)
final_score_gain    : ranked_final - baseline_final (positive = improvement)
"""

import csv
import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"
COMPARE_ENDPOINT = f"{API_BASE_URL}/compare"

DATA_FILE = os.path.join("data", "eval_prompts.csv")
RESULTS_CSV = os.path.join("results", "evaluation_results.csv")
SUMMARY_JSON = os.path.join("results", "summary.json")

# Request parameters — keep beams low for faster evaluation
DEFAULT_BEAMS = 3
DEFAULT_MAX_TOKENS = 50
DEFAULT_ALPHA = 0.7

# Seconds to wait between requests (be kind to your CPU)
REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_server_health():
    """
    Verify the FastAPI server is reachable before starting the benchmark.
    Prints a helpful message and exits if the server is not running.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=5)
        resp.raise_for_status()
        print(f"Server is running: {resp.json().get('service', 'OK')}")
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] FastAPI server is not running.\n"
            "Start it using:\n"
            "    uvicorn app.main:app --reload\n"
            "Then re-run this script."
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"[ERROR] Server returned an error: {exc}")
        sys.exit(1)


def load_prompts(filepath: str) -> list[dict]:
    """
    Read prompts from a CSV file with columns: prompt, category.

    Returns a list of dicts: [{"prompt": "...", "category": "..."}, ...]
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Prompt file not found: {filepath}")
        sys.exit(1)

    prompts = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("prompt", "").strip():
                prompts.append({
                    "prompt": row["prompt"].strip(),
                    "category": row.get("category", "unknown").strip(),
                })

    print(f"Loaded {len(prompts)} prompts from {filepath}")
    return prompts


def call_compare(prompt: str):
    """
    Call POST /compare for a single prompt.

    Returns the parsed JSON response, or None if the request failed
    (e.g. prompt was blocked, server error, timeout).
    """
    payload = {
        "text": prompt,
        "beams": DEFAULT_BEAMS,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "alpha": DEFAULT_ALPHA,
    }
    try:
        resp = requests.post(COMPARE_ENDPOINT, json=payload, timeout=120)

        # Blocked prompts return 400 — log and skip rather than crash
        if resp.status_code == 400:
            detail = resp.json().get("detail", {})
            if isinstance(detail, dict) and detail.get("status") == "blocked":
                print(f"  [BLOCKED] prompt_risk={detail.get('prompt_risk', '?')}")
                return None

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        print("  [TIMEOUT] Request took too long — skipping.")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"  [ERROR] Request failed: {exc}")
        return None


def flatten_result(prompt: str, category: str, data: dict) -> dict:
    """
    Flatten the /compare response into a single CSV-friendly row.

    Extracts the key metrics from baseline_scores and safety_ranked_scores
    plus the improvement block.
    """
    bs = data["baseline_scores"]
    sr = data["safety_ranked_scores"]
    imp = data["improvement"]

    return {
        "prompt": prompt,
        "category": category,
        # Baseline metrics
        "baseline_toxicity": bs["toxicity_score"],
        "baseline_bias": bs["bias_score"],
        "baseline_final_score": bs["final_score"],
        # Safety-ranked metrics
        "ranked_toxicity": sr["toxicity_score"],
        "ranked_bias": sr["bias_score"],
        "ranked_final_score": sr["final_score"],
        # Improvement
        "toxicity_safety_gain": imp["toxicity_safety_gain"],
        "bias_safety_gain": imp["bias_safety_gain"],
        "final_score_gain": imp["final_score_gain"],
        # Outputs (for qualitative review)
        "baseline_output": data["baseline_output"],
        "safety_ranked_output": data["safety_ranked_output"],
    }


def compute_averages(rows: list[dict]) -> dict:
    """
    Compute mean values for all numeric columns, overall and per category.

    Returns a dict structured as:
    {
        "overall": { "avg_baseline_toxicity": ..., ... },
        "by_category": {
            "normal": { ... },
            "toxic": { ... },
            ...
        },
        "total_prompts": N,
        "blocked_prompts": M,
    }
    """
    numeric_cols = [
        "baseline_toxicity", "baseline_bias", "baseline_final_score",
        "ranked_toxicity", "ranked_bias", "ranked_final_score",
        "toxicity_safety_gain", "bias_safety_gain", "final_score_gain",
    ]

    def mean_of(data: list[dict], cols: list[str]) -> dict:
        if not data:
            return {f"avg_{c}": None for c in cols}
        return {
            f"avg_{c}": round(sum(r[c] for r in data) / len(data), 4)
            for c in cols
        }

    # Group by category
    categories: dict[str, list[dict]] = {}
    for row in rows:
        cat = row["category"]
        categories.setdefault(cat, []).append(row)

    return {
        "total_prompts_attempted": None,   # filled in by caller
        "total_prompts_evaluated": len(rows),
        "overall": mean_of(rows, numeric_cols),
        "by_category": {
            cat: {"count": len(cat_rows), **mean_of(cat_rows, numeric_cols)}
            for cat, cat_rows in categories.items()
        },
    }


def save_csv(rows: list[dict], filepath: str):
    """Write result rows to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        print(f"[WARNING] No rows to write to {filepath}")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {filepath}")


def save_json(data: dict, filepath: str):
    """Write summary dict to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved summary → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EthicalGuard — Benchmark Evaluation")
    print("=" * 60)

    # 1. Verify server is up
    check_server_health()

    # 2. Load prompts
    prompts = load_prompts(DATA_FILE)
    total_attempted = len(prompts)

    # 3. Run evaluation
    result_rows: list[dict] = []
    blocked_count = 0

    for i, item in enumerate(prompts, start=1):
        prompt = item["prompt"]
        category = item["category"]
        print(f"\n[{i}/{total_attempted}] [{category}] {prompt[:60]}...")

        response = call_compare(prompt)

        if response is None:
            # Blocked or failed — count it but don't add to results
            blocked_count += 1
            continue

        row = flatten_result(prompt, category, response)
        result_rows.append(row)

        # Show a quick summary line
        print(
            f"  baseline={row['baseline_final_score']:.3f}  "
            f"ranked={row['ranked_final_score']:.3f}  "
            f"gain={row['final_score_gain']:+.3f}  "
            f"tox_reduction={row['toxicity_reduction']:+.3f}"
        )

        time.sleep(REQUEST_DELAY)

    # 4. Save per-prompt results
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {len(result_rows)} evaluated, {blocked_count} blocked/failed")
    save_csv(result_rows, RESULTS_CSV)

    # 5. Compute and save summary
    summary = compute_averages(result_rows)
    summary["total_prompts_attempted"] = total_attempted
    summary["blocked_or_failed"] = blocked_count
    save_json(summary, SUMMARY_JSON)

    # 6. Print headline numbers
    ov = summary["overall"]
    print("\n--- Overall Averages ---")
    print(f"  Baseline  toxicity : {ov.get('avg_baseline_toxicity', 'N/A')}")
    print(f"  Ranked    toxicity : {ov.get('avg_ranked_toxicity', 'N/A')}")
    print(f"  Toxicity safety gain : {ov.get('avg_toxicity_safety_gain', 'N/A')}")
    print(f"  Bias safety gain     : {ov.get('avg_bias_safety_gain', 'N/A')}")
    print(f"  Baseline  final    : {ov.get('avg_baseline_final_score', 'N/A')}")
    print(f"  Ranked    final    : {ov.get('avg_ranked_final_score', 'N/A')}")
    print(f"  Final score gain   : {ov.get('avg_final_score_gain', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
