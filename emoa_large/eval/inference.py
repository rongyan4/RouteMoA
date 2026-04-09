#!/usr/bin/env python3
"""
inference.py — Standalone inference script for the benchmark datasets.

Reads `benchmark_questions.json`, calls an OpenAI-compatible API for each item,
and writes predictions to `predictions.json`.

Usage:
    python inference.py \
        --base_url https://api.openai.com/v1 \
        --api_key  YOUR_API_KEY \
        --model    gpt-4o \
        [--input   benchmark_questions.json] \
        [--output  predictions.json] \
        [--workers 4]

The script supports resuming: items whose `model_response` is already present in
the output file are skipped on re-run.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run inference on benchmark_questions.json")
    p.add_argument("--base_url", required=True,
                   help="Base URL of the OpenAI-compatible API endpoint, "
                        "e.g. https://api.openai.com/v1")
    p.add_argument("--api_key", required=True,
                   help="API key for the endpoint")
    p.add_argument("--model", required=True,
                   help="Model name to use, e.g. gpt-4o")
    p.add_argument("--input", default="benchmark_questions.json",
                   help="Path to the benchmark questions file (default: benchmark_questions.json)")
    p.add_argument("--output", default="predictions.json",
                   help="Path to write predictions (default: predictions.json)")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel API workers (default: 4)")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Max retries per item on API error (default: 3)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_predictions(path: str) -> dict:
    """Returns dict keyed by (category, dataset, item_id) → model_response."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    done = {}
    for category, datasets in data.get("data", {}).items():
        for dataset, ds_info in datasets.items():
            for item in ds_info.get("items", []):
                if item.get("model_response") is not None:
                    key = (category, dataset, item["item_id"])
                    done[key] = item["model_response"]
    return done


def call_api(client: OpenAI, model: str, prompt: str,
             max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """Call the chat completions endpoint with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8192,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"  [retry {attempt+1}/{max_retries-1}] error: {exc}; waiting {wait}s …",
                      flush=True)
                time.sleep(wait)
            else:
                print(f"  [FAILED after {max_retries} attempts] {exc}", flush=True)
                return f"[ERROR] {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load benchmark questions
    print(f"Loading benchmark questions from: {args.input}")
    benchmark = load_benchmark(args.input)

    # Load already-done predictions (for resuming)
    print(f"Checking for existing predictions in: {args.output}")
    existing = load_existing_predictions(args.output)
    print(f"  Found {len(existing)} already-completed items — will skip these.")

    # Build a flat list of tasks
    tasks = []
    for category, datasets in benchmark["data"].items():
        for dataset, ds_info in datasets.items():
            for item in ds_info["items"]:
                key = (category, dataset, item["item_id"])
                if key not in existing:
                    tasks.append((category, dataset, item))
    print(f"Items to process: {len(tasks)}")

    # Deep-copy the benchmark structure to build output
    output = json.loads(json.dumps(benchmark))
    # Inject any already-done responses
    for category, datasets in output["data"].items():
        for dataset, ds_info in datasets.items():
            for item in ds_info["items"]:
                key = (category, dataset, item["item_id"])
                if key in existing:
                    item["model_response"] = existing[key]

    # Add inference metadata to output
    output["inference_metadata"] = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    # Initialize the client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Build a lookup for fast writes: (category, dataset, item_id) → item dict
    item_lookup = {}
    for category, datasets in output["data"].items():
        for dataset, ds_info in datasets.items():
            for item in ds_info["items"]:
                item_lookup[(category, dataset, item["item_id"])] = item

    write_lock = Lock()
    completed_count = [0]
    total = len(tasks)

    def save_output():
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def process_item(task):
        category, dataset, item = task
        prompt = item["full_prompt"]
        response = call_api(client, args.model, prompt, args.max_retries)
        with write_lock:
            item_lookup[(category, dataset, item["item_id"])]["model_response"] = response
            completed_count[0] += 1
            count = completed_count[0]
            if count % 10 == 0 or count == total:
                save_output()
                print(f"  [{count}/{total}] saved checkpoint", flush=True)
            else:
                print(f"  [{count}/{total}] {dataset} / {item['item_id']}: done", flush=True)

    if total > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_item, t): t for t in tasks}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    task = futures[fut]
                    print(f"  [EXCEPTION] {task[1]} / {task[2]['item_id']}: {exc}", flush=True)

        # Final save
        save_output()

    print(f"\nDone! Predictions written to: {args.output}")

    # Print summary stats
    total_items = sum(
        len(ds_info["items"])
        for datasets in output["data"].values()
        for ds_info in datasets.values()
    )
    done_items = sum(
        1
        for datasets in output["data"].values()
        for ds_info in datasets.values()
        for item in ds_info["items"]
        if item.get("model_response") is not None
    )
    print(f"Total items: {total_items}  |  With responses: {done_items}")


if __name__ == "__main__":
    main()
