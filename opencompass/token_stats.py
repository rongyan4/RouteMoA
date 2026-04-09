#!/usr/bin/env python
# token_stats.py
"""
Compute average token usage and per-rank counts (top-5) for each model
from one or more JSON files that share a common filename prefix.

The user only needs to change BASE_DIR and FILE_PREFIX below.
No other dependencies beyond the Python standard library.
"""

from pathlib import Path
import json
from collections import defaultdict, Counter

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
BASE_DIR: Path = Path("/tos-bjml-llm-dev/liyining/wangjize/opencompass/outputs/emoa/20250722_162446/predictions/emoa-mmlu-mbpp-race")           # ← folder to search
FILE_PREFIXES: list[str] = [                            # ← filenames start‑with
    "ARC-c_",
    "gsm8k_",
    "lukaemon_",
    "math_",
    "race-high_",
    "sanitized_mbpp_",
]
# ─── OUTPUT CONFIGURATION ───────────────────────────────────────────────────
OUTPUT_MD_FILE: Path = Path("/tos-bjml-llm-dev/liyining/wangjize/opencompass/token_result/emoa-mmlu-mbpp-race.md")  # <- Modifiable, write all results to this Markdown file
# ───────────────────────────────────────────────────────────────────────────────

def load_examples(base_dir: Path, prefix: str):
    """Return a list of all sample dicts merged from every JSON file
    whose filename begins with `prefix` inside `base_dir`."""
    examples = []
    for file in sorted(base_dir.glob(f"{prefix}*.json")):
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        # each data is a dict whose values are the examples
        examples.extend(data.values())
    if not examples:
        raise FileNotFoundError(f"No *.json files beginning with '{prefix}' found in {base_dir}")
    return examples


def aggregate_statistics(samples):
    """Return (avg_prompt, avg_completion, rank_prompt, rank_completion)."""
    # token sums per model
    prompt_sum = Counter()
    comp_sum   = Counter()
    # how many samples we actually saw
    n_samples = len(samples)

    # rank-count dictionaries: rank 1-5 -> counter per model
    rank_prompt = {k: Counter() for k in range(1, 6)}
    rank_comp   = {k: Counter() for k in range(1, 6)}

    for sample in samples:
        bill = sample["bill"]
        p_tok = bill["prompt_tokens"]
        c_tok = bill["completion_tokens"]

        # update sums
        prompt_sum.update(p_tok)
        comp_sum.update(c_tok)

        # ---------- ranking helpers ----------
        def update_ranks(tok_dict, rank_counter_dict):
            # sort models by token count (DESC). Ties keep file order.
            ranked = sorted(tok_dict.items(), key=lambda kv: (-kv[1], kv[0]))
            for idx, (model, _) in enumerate(ranked[:5]):  # only ranks 1-5
                rank_counter_dict[idx + 1][model] += 1

        update_ranks(p_tok, rank_prompt)
        update_ranks(c_tok, rank_comp)

    # convert sums to averages
    avg_prompt = {m: total / n_samples for m, total in prompt_sum.items()}
    avg_comp   = {m: total / n_samples for m, total in comp_sum.items()}

    return avg_prompt, avg_comp, rank_prompt, rank_comp


# Add MODEL_ALIASES after existing imports at the beginning (for sharing)
MODEL_ALIASES = {
    "ContactDoctor/Bio-Medical-Llama-3-8B": "bio",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Coder",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Math",
    "google/gemma-2-9b-it": "gemma",
    "mistralai/Ministral-8B-Instruct-2410": "ministral",
}

# --- Add a generic table constructor function --------------------------------------------
def _make_md_table(headers: list[str],
                   rows: list[list[str]],
                   aligns: list[str]) -> str:
    """
    headers: Column names
    rows   : List of strings for each row
    aligns : 'left' / 'right' Specify alignment
    """
    # 1. Calculate widths
    ncols = len(headers)
    widths = [len(headers[i]) for i in range(ncols)]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # 2. Construct formatter
    def fmt(i: int, text: str) -> str:
        return text.ljust(widths[i]) if aligns[i] == "left" else text.rjust(widths[i])

    # 3. Generate rows
    header_line = "| " + " | ".join(fmt(i, h) for i, h in enumerate(headers)) + " |"
    # Separator row: Left align with pure "-", right align with ":" at the end (Markdown syntax)
    sep_cells = []
    for i in range(ncols):
        dash = "-" * widths[i]
        sep_cells.append(dash if aligns[i] == "left" else dash + ":")
    sep_line = "| " + " | ".join(sep_cells) + " |"

    data_lines = [
        "| " + " | ".join(fmt(i, cell) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line, *data_lines])
# ────────────────────────────────────────────────────────────────────────────

def format_md_table_avg(avg_prompt, avg_comp):
    """Return *visually aligned* markdown of average token counts per model."""
    headers = ["Model", "Avg Prompt Tokens", "Avg Completion Tokens"]
    aligns  = ["left", "right", "right"]

    rows = [
        [
            MODEL_ALIASES.get(model, model),
            f"{avg_prompt.get(model, 0):.2f}",
            f"{avg_comp.get(model, 0):.2f}",
        ]
        for model in sorted(set(avg_prompt) | set(avg_comp))
    ]
    return _make_md_table(headers, rows, aligns)


def format_md_table_ranks(rank_dict, title):
    """Return *visually aligned* markdown table for rank stats."""
    headers = ["Model", "#Rank-1", "#Rank-2", "#Rank-3", "#Rank-4", "#Rank-5"]
    aligns  = ["left"] + ["right"] * 5

    all_models = {m for r in rank_dict.values() for m in r}
    rows = []
    for m in sorted(all_models):
        short_name = MODEL_ALIASES.get(m, m)
        counts = [str(rank_dict[k].get(m, 0)) for k in range(1, 6)]
        rows.append([short_name, *counts])

    table_body = _make_md_table(headers, rows, aligns)
    return f"\n\n### {title}\n{table_body}"


def main():
    output_lines: list[str] = []

    for prefix in FILE_PREFIXES:
        try:
            samples = load_examples(BASE_DIR, prefix)
        except FileNotFoundError as e:
            output_lines.append(f"\n⚠️  {e}")
            continue

        output_lines.append(f"\n\n## Statistics for prefix '{prefix}'\n")
        avg_p, avg_c, r_p, r_c = aggregate_statistics(samples)

        output_lines.append("\n\n## Average Token Usage per Model\n")
        output_lines.append(format_md_table_avg(avg_p, avg_c))

        output_lines.append(format_md_table_ranks(r_p, "Prompt‑Token Rank Counts"))
        output_lines.append(format_md_table_ranks(r_c, "Completion‑Token Rank Counts"))

    # Output to console
    result_text = "".join(output_lines)
    print(result_text)

    # Write to Markdown file
    try:
        OUTPUT_MD_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"\n✅ Results saved to {OUTPUT_MD_FILE}")
    except Exception as e:
        print(f"\n⚠️ Failed to write results to {OUTPUT_MD_FILE}: {e}")


if __name__ == "__main__":
    main()