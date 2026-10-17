"""
RunPod Serverless Worker — Sentiment Analysis Handler

Demonstrates the full RunPod handler contract: module-level model
loading, input validation, structured JSON output, and local testing
via the SDK's built-in test mode.
"""

import runpod

# ── Model loads at module level: once per worker, reused across all jobs ──

from transformers import pipeline

CLASSIFIER = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)


def handler(job):
    """
    Accepts: {"input": {"text": "...", "top_k": 2}}
    Returns: {"labels": [...], "input_length": N}
    """
    job_input = job["input"]

    text = job_input.get("text")
    if not text:
        raise ValueError("Missing required field: 'text'")

    top_k = job_input.get("top_k", 2)

    results = CLASSIFIER(text[:512], top_k=top_k)

    return {
        "labels": [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in results
        ],
        "input_length": len(text),
    }


runpod.serverless.start({"handler": handler})
