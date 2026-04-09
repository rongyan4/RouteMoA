#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation script for the benchmark predictions.

Reads a predictions file (produced by inference.py) and benchmark_questions.json,
then computes per-dataset scores and category averages.

Usage:
    python evaluate.py \
        --predictions  predictions.json \
        [--benchmark   benchmark_questions.json] \
        [--output      eval_results.json] \
        [--judge_base_url  https://api.openai.com/v1] \
        [--judge_api_key   YOUR_OPENAI_KEY] \
        [--judge_model     gpt-4o]

Notes on dependencies:
  • RougeMetric  → pip install lawrouge
  • GEC_f1       → requires m2score (levenshtein), pkuseg (Chinese), nltk (English)
                   pip install pkuseg nltk
                   git clone https://github.com/nusnlp/m2scorer  (add to PYTHONPATH)
  • ModelMetric  → requires an OpenAI-compatible API (pass --judge_* flags)
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate predictions against benchmark references")
    p.add_argument("--predictions", default="predictions.json",
                   help="Path to predictions JSON (output of inference.py)")
    p.add_argument("--benchmark", default="benchmark_questions.json",
                   help="Path to benchmark questions JSON")
    p.add_argument("--output", default="eval_results.json",
                   help="Where to write evaluation results")
    p.add_argument("--judge_base_url", default=None,
                   help="Base URL of OpenAI-compatible API for LLM judge (ModelMetric datasets)")
    p.add_argument("--judge_api_key", default=None,
                   help="API key for the LLM judge")
    p.add_argument("--judge_model", default="gpt-4o",
                   help="Model name for the LLM judge (default: gpt-4o)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for ModelMetric calls (default: 4)")
    return p.parse_args()


# ===========================================================================
# Metric helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.  Accuracy — keyword maps + match modes
# ---------------------------------------------------------------------------

KEYWORD_MAPS: Dict[str, Dict[str, str]] = {
    "lcqmc": {"是": "1", "否": "0", "相同": "1", "不同": "0",
               "false": "0", "yes": "1", "Yes": "1", "1": "1", "0": "0"},
    "ocnli": {"矛盾": "contradiction", "蕴含": "entailment", "中立": "neutral"},
    "sst2":  {"Positive": "Positive", "positive": "Positive",
               "Negative": "Negative", "negative": "Negative",
               "积极": "Positive", "消极": "Negative",
               "Pos": "Positive", "Neg": "Negative",
               "pos": "Positive", "neg": "Negative"},
    "cola":  {"Yes": "Yes", "yes": "Yes", "No": "No", "no": "No"},
    "mrpc":  {"yes": "Yes", "Yes": "Yes", "no": "No", "No": "No"},
    "qqp":   {"yes": "Yes", "Yes": "Yes", "no": "No", "No": "No"},
    "sts_b": {"yes": "Yes", "Yes": "Yes", "no": "No", "No": "No"},
    "ag_news": {"World": "World", "Sports": "Sports", "Business": "Business",
                "Science and Technology": "Science and Technology",
                "Science": "Science and Technology",
                "Technology": "Science and Technology"},
    "qnli":  {"yes": "entailment", "Yes": "entailment",
               "no": "not_entailment", "No": "not_entailment"},
    "race":  {"A": "A", "B": "B", "C": "C", "D": "D"},
    "story_cloze_test": {"A": "A", "B": "B"},
    "cluewsc2020": {"A": "true", "B": "false",
                    "是。": "true", "是的": "true",
                    "不是": "false", "否": "false"},
    "truthful_qa": {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E"},
}

# Datasets that use a NumberParser instead of keyword matching
NUMBER_PARSER_MODE = {
    "bigmath":          "last",   # last number in response
    "gsm8k_test_100":   "last3",  # last number among last-3 lines
    "GAOKAO-2023_Math_en": "last4",  # last number among last-4 lines
}

# Match modes for accuracy datasets
MATCH_MODES = {
    "lcqmc":            "answer_include_ref",
    "ocnli":            "exact_match",
    "sst2":             "exact_match",
    "cola":             "exact_match",
    "mrpc":             "exact_match",
    "qqp":              "exact_match",
    "sts_b":            "exact_match",
    "ag_news":          "exact_match",
    "qnli":             "exact_match",
    "chid_baidu":       "answer_include_any_ref",
    "webqa":            "answer_include_ref",
    "c3":               "answer_include_any_ref",
    "cmrc":             "answer_include_ref",
    "race":             "exact_match",
    "story_cloze_test": "exact_match",
    "cluewsc2020":      "exact_match",
    "winograd_wsc":     "any_include",
    "truthful_qa":      "exact_match",
    "bigmath":          "exact_match",
    "gsm8k_test_100":   "answer_include_ref",
    "GAOKAO-2023_Math_en": "answer_include_ref",
    "word_manipulation_v2":  "answer_include_ref",
    "nlpcc2017_task2":  "answer_include_any_ref",
}


def apply_keyword_map(text: str, kmap: Dict[str, str]) -> Optional[str]:
    """Return the first matching keyword-mapped value, or None."""
    for keyword, mapped in kmap.items():
        if keyword in text:
            return mapped
    return None


def _extract_number(text: str) -> Optional[str]:
    """Extract all numbers (integers or decimals) from text."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums


def parse_number(answer: str, mode: str) -> Optional[str]:
    """Extract a number from the model answer per the given mode."""
    if mode == "last":
        nums = _extract_number(answer)
        return nums[-1] if nums else None
    elif mode == "last3":
        lines = [l.strip() for l in answer.strip().splitlines() if l.strip()]
        chunk = " ".join(lines[-3:]) if len(lines) >= 3 else answer
        nums = _extract_number(chunk)
        return nums[-1] if nums else None
    elif mode == "last4":
        lines = [l.strip() for l in answer.strip().splitlines() if l.strip()]
        chunk = " ".join(lines[-4:]) if len(lines) >= 4 else answer
        nums = _extract_number(chunk)
        return nums[-1] if nums else None
    return None


def accuracy_match(pred: str, ref: Any, mode: str) -> float:
    """Return 1.0 if pred matches ref according to mode, else 0.0."""
    pred_s = str(pred).strip() if pred is not None else ""
    if isinstance(ref, list):
        refs = [str(r).strip() for r in ref]
    else:
        refs = [str(ref).strip()]

    if mode == "exact_match":
        return float(pred_s in refs)
    elif mode == "answer_include_ref":
        return float(any(r in pred_s for r in refs))
    elif mode == "ref_include_answer":
        return float(any(pred_s in r for r in refs))
    elif mode == "answer_include_any_ref":
        return float(any(r in pred_s for r in refs))
    elif mode == "any_include":
        # lowercased: either answer includes ref or ref includes answer
        pl = pred_s.lower()
        return float(any(r.lower() in pl or pl in r.lower() for r in refs))
    else:
        return float(pred_s in refs)


def eval_accuracy(dataset: str, answer: str, ref: Any) -> float:
    """Compute accuracy for a single item."""
    kmap = KEYWORD_MAPS.get(dataset)
    mode = MATCH_MODES.get(dataset, "exact_match")

    # Number parser datasets
    if dataset in NUMBER_PARSER_MODE:
        num_mode = NUMBER_PARSER_MODE[dataset]
        parsed = parse_number(answer, num_mode)
        if parsed is None:
            return 0.0
        return accuracy_match(parsed, ref, mode)

    # Keyword map datasets
    if kmap:
        parsed = apply_keyword_map(answer, kmap)
        if parsed is None:
            return 0.0
        return accuracy_match(parsed, ref, mode)

    # Direct (no keyword map, no number parser)
    return accuracy_match(answer, ref, mode)


# ---------------------------------------------------------------------------
# 2.  Recall (MSRA NER)
# ---------------------------------------------------------------------------

def eval_recall(answer: str, ref: str) -> float:
    """Compute recall: fraction of ref entities found in answer."""
    delimiter = re.compile(r"[;,，；]")
    entities = [e.strip() for e in delimiter.split(ref) if e.strip()]
    if not entities:
        return 0.0
    matched = sum(1 for e in entities if e in answer)
    return matched / len(entities)


# ---------------------------------------------------------------------------
# 3.  QA_f1 (character-level F1 for Chinese)
# ---------------------------------------------------------------------------

def _normalize_cn(s: str) -> List[str]:
    punctuation = r"""!"#$%&'()*+_,./:;<>=?@[]\^_`{}|~！￥……（）——【】'：；，《》"。，、？"""
    exclude = set(punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    s = " ".join(s.split())
    s = s.lower()
    return list(s)  # character-level tokens


def eval_qa_f1(answer: str, ref: str, lang: str = "cn") -> float:
    if lang == "cn":
        gold_toks = _normalize_cn(ref)
        pred_toks = _normalize_cn(answer)
    else:
        import string
        def normalize_en(s):
            s = re.sub(r"\b(a|an|the)\b", " ", s, flags=re.UNICODE)
            s = "".join(ch for ch in s if ch not in set(string.punctuation))
            return " ".join(s.split()).lower()
        gold_toks = normalize_en(ref).split()
        pred_toks = normalize_en(answer).split()

    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# 4.  RougeMetric (lawrouge, Chinese)
# ---------------------------------------------------------------------------

def eval_rouge(answer: str, ref: str) -> Dict[str, float]:
    """Return {'rouge_1': r1_recall, 'rouge_2': r2_recall, 'rouge_l': rl_f1}."""
    try:
        from lawrouge import Rouge
    except ImportError:
        print("[WARNING] lawrouge not installed. pip install lawrouge", file=sys.stderr)
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    rouge = Rouge(isChinese=True)
    if not answer.strip() or not ref.strip():
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    try:
        scores = rouge.get_scores(answer, ref)[0]
        return {
            "rouge_1": scores["rouge-1"]["r"],
            "rouge_2": scores["rouge-2"]["r"],
            "rouge_l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        print(f"[WARNING] RougeMetric error: {e}", file=sys.stderr)
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


# ---------------------------------------------------------------------------
# 5.  GEC_f1 (m2score, grammar error correction)
# ---------------------------------------------------------------------------

# Regex patterns for extracting the corrected sentence from model output
GEC_EXTRACT_PATTERNS = {
    "nlpcc2018_task2": re.compile(
        r"(?:修改后的句子：|修改后的句子:|修改后：|修改后:|)(.+?)(?:</s>|$)",
        re.DOTALL
    ),
    "conll2014": re.compile(
        r"(?:The sentence should be:|The modified sentence would be:|"
        r"The correct sentence would be:|The modified sentence is:|"
        r"The correct sentence should be:|Revised sentence:|Modified sentence:|)(.+?)(?:</s>|$)",
        re.DOTALL
    ),
}


def _extract_gec_sentence(dataset: str, answer: str) -> str:
    """Extract the corrected sentence from a model response."""
    pattern = GEC_EXTRACT_PATTERNS.get(dataset)
    if pattern:
        m = pattern.search(answer)
        if m:
            return m.group(1).strip()
    return answer.strip()


def _is_chinese(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


def _tokenize_gec(text: str, is_cn: bool) -> List[str]:
    if is_cn:
        try:
            import pkuseg
            seg = pkuseg.pkuseg()
            return seg.cut(text)
        except ImportError:
            return list(text)
    else:
        try:
            import nltk
            return nltk.word_tokenize(text)
        except Exception:
            return text.split()


def eval_gec_f1(dataset: str, answer: str, ref: Any,
                timeout_secs: int = 120) -> float:
    """
    Compute GEC F1 using m2score levenshtein.

    `ref` is expected to be the parsed m2 edit list (list of strings) as stored in
    benchmark_questions.json, or a plain corrected-sentence string as fallback.

    Returns the F1 score in [0, 1].
    """
    extracted = _extract_gec_sentence(dataset, answer)

    # Try to use the proper m2score approach
    try:
        from lib.m2score.levenshtein import batch_multi_pre_rec_f1
    except ImportError:
        print("[WARNING] m2score (lib.m2score.levenshtein) not found. "
              "Falling back to character-level F1.", file=sys.stderr)
        ref_str = ref[0] if isinstance(ref, list) else str(ref)
        return eval_qa_f1(extracted, ref_str, lang="cn" if _is_chinese(ref_str) else "en")

    is_cn = _is_chinese(extracted)

    result_holder = [None]
    error_holder = [None]

    def run():
        try:
            tokens_pred = _tokenize_gec(extracted, is_cn)
            if isinstance(ref, list):
                ref_edits = ref  # list of edit annotations
            else:
                ref_edits = [str(ref)]
            p, r, f = batch_multi_pre_rec_f1(
                [tokens_pred], [ref_edits], 0.5
            )
            result_holder[0] = f
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        print(f"[WARNING] GEC_f1 timed out for dataset={dataset}", file=sys.stderr)
        return 0.0
    if error_holder[0]:
        print(f"[WARNING] GEC_f1 error: {error_holder[0]}", file=sys.stderr)
        return 0.0
    return result_holder[0] if result_holder[0] is not None else 0.0


# ---------------------------------------------------------------------------
# 6.  ModelMetric (LLM judge, gpt-4o)
# ---------------------------------------------------------------------------

JUDGE_PROMPT_MATH = """\
你是一个擅长评判数学题质量的助手。
请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是专业能力，因此你需要从下面的几个维度对回答进行评估:
1. 逻辑正确性: 回答中提供的解题过程在逻辑上是否正确无误，提供的答案是否与标准正确答案一致。
2. 满足用户需求: 回答是否满足了用户提出问题的目的和需求，是否对问题进行了全面而恰当的回应。
我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：
1. 将AI助手的答案与参考答案进行比较，指出AI助手的答案有哪些不足，并进一步解释。
2. 从不同维度对AI助手的答案进行评价，在每个维度的评价之后，给每一个维度一个0～4的分数。
3. 最后，综合每个维度的评估，对AI助手的回答给出一个0~4的综合分数。
4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。

- 逻辑正确性评分规则：
0分-完全错误，AI助手回答的答案解题过程中的逻辑推理完全不符合数学规则，存在严重逻辑错误，结论与标准正确答案完全不一致。
1分-较差，AI助手回答的答案解题过程中的逻辑推理较为混乱，存在较多逻辑错误，结论与标准正确答案不一致。
2分-部分正确，AI助手回答的答案解题过程中的逻辑推理存在少量逻辑错误，结论与标准正确答案部分一致。
3分-较好，AI助手回答的答案解题过程中的逻辑推理较为严谨，基本无逻辑错误，结论与标准正确答案基本一致。
4分-最高分，AI助手回答的答案解题过程中的逻辑推理非常严谨，无逻辑错误，结论与标准正确答案完全一致。
- 满足用户需求评分规则：
0分-完全错误，AI助手的回答完全不能满足用户需求，或存在多处错误，无法解决用户提出的问题。
1分-较差，AI助手能理解用户需求，但回答内容较差，存在多处错误或不完整，无法有效解决用户问题。
2分-基本满足，AI助手能理解用户需求，回答内容基本能满足用户需求，但存在少量错误或不完整之处。
3分-较好，AI助手的回答能较好地满足用户需求，回答内容质量较高，基本无错误，能够有效解决用户问题。
4分-最高分，AI助手的回答非常完美地满足了用户需求，回答内容质量极高，无任何错误，能够非常有效地解决用户问题。
- 综合得分评分规则：
0分：回答在所有维度上表现极差，无任何可取之处
1分：回答在多数维度上表现不佳，整体质量较差
2分：回答在部分维度上表现良好，但整体质量一般
3分：回答在大部分维度上表现良好，整体质量较高
4分：回答在所有维度上表现优秀，整体质量非常高

请记住，你必须在你打分前进行评价和解释。在你对每个维度的解释之后，需要加上对该维度的打分。最后，在你回答的末尾以 'Score: #' 的格式输出综合评分，并确保你的打分结果是整数：

<|用户问题的开始|>
{problem}
<|用户问题的结束|>

<|标准正确答案的开始|>
{solution}
<|标准正确答案的结束|>

<|AI助手回答的开始|>
{answer}
<|AI助手回答的结束|>"""

# Datasets that use ModelMetric
MODEL_METRIC_DATASETS = {"geometry", "prealgebra", "precalculus"}


def _parse_judge_score(text: str, max_score: int = 4) -> Optional[float]:
    """Parse 'Score: N' from judge output, fallback to first number ≤ max_score."""
    m = re.search(r"Score:\s*(\d+)", text)
    if m:
        return float(m.group(1))
    # fallback: first integer ≤ max_score
    nums = re.findall(r"\d+", text)
    for n in nums:
        if int(n) <= max_score:
            return float(n)
    return None


def eval_model_metric(item: dict, judge_client, judge_model: str,
                       max_retries: int = 3) -> float:
    """
    Call the LLM judge to score a math answer.
    Returns normalized score in [0, 1].
    """
    problem = item["question"]
    solution = item["ground_truth"]
    answer = item.get("model_response", "")

    prompt = JUDGE_PROMPT_MATH.format(
        problem=problem, solution=solution, answer=answer
    )

    for attempt in range(max_retries):
        try:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            text = response.choices[0].message.content or ""
            score = _parse_judge_score(text, max_score=4)
            if score is not None:
                return min(score, 4.0) / 4.0
            print(f"[WARNING] Could not parse score from judge response: {text[:200]}",
                  file=sys.stderr)
            return 0.0
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"[ERROR] Judge API failed: {e}", file=sys.stderr)
                return 0.0
    return 0.0


# ===========================================================================
# Dataset dispatch
# ===========================================================================

# Which metric class each dataset uses
DATASET_METRIC_TYPE = {
    # Language Understanding
    "lcqmc":            "accuracy",
    "ocnli":            "accuracy",
    "sst2":             "accuracy",
    "cola":             "accuracy",
    "mrpc":             "accuracy",
    "msra":             "recall",
    "qqp":              "accuracy",
    "sts_b":            "accuracy",
    "ag_news":          "accuracy",
    "qnli":             "accuracy",
    "chid_baidu":       "accuracy",
    # Reading & QA
    "webqa":            "accuracy+qa_f1",
    "c3":               "accuracy",
    "cmrc":             "accuracy+qa_f1",
    "race":             "accuracy",
    "story_cloze_test": "accuracy",
    # Logic Reasoning
    "cluewsc2020":      "accuracy",
    "winograd_wsc":     "accuracy",
    "truthful_qa":      "accuracy",
    # Math Reasoning
    "bigmath":          "accuracy",
    "gsm8k_test_100":   "accuracy",
    "GAOKAO-2023_Math_en": "accuracy",
    "geometry":         "model_metric",
    "prealgebra":       "model_metric",
    "precalculus":      "model_metric",
    # Language Generation
    "word_manipulation_v2": "accuracy",
    "nlpcc2017_task2":  "accuracy",
    "lcsts":            "rouge",
    "nlpcc2018_task2":  "gec_f1",
    "conll2014":        "gec_f1",
}


def eval_item(dataset: str, item: dict, judge_client=None, judge_model: str = "gpt-4o") -> dict:
    """
    Evaluate a single item. Returns a dict of metric_name → score.
    """
    answer = item.get("model_response", "")
    ref = item.get("ground_truth", "")
    mtype = DATASET_METRIC_TYPE.get(dataset, "accuracy")

    if answer is None:
        answer = ""

    if mtype == "accuracy":
        return {"accuracy": eval_accuracy(dataset, answer, ref)}

    elif mtype == "recall":
        return {"recall": eval_recall(answer, ref)}

    elif mtype == "accuracy+qa_f1":
        acc = eval_accuracy(dataset, answer, ref)
        f1 = eval_qa_f1(answer, ref, lang="cn")
        return {"accuracy": acc, "qa_f1": f1}

    elif mtype == "rouge":
        scores = eval_rouge(answer, ref)
        return scores  # keys: rouge_1, rouge_2, rouge_l

    elif mtype == "gec_f1":
        f1 = eval_gec_f1(dataset, answer, ref)
        return {"gec_f1": f1}

    elif mtype == "model_metric":
        if judge_client is None:
            print(f"[WARNING] No judge client configured; skipping ModelMetric for {dataset}",
                  file=sys.stderr)
            return {"model_score": None}
        score = eval_model_metric(item, judge_client, judge_model)
        return {"model_score": score}

    else:
        return {"accuracy": eval_accuracy(dataset, answer, ref)}


# ===========================================================================
# Aggregation helpers
# ===========================================================================

def mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def primary_metric_name(dataset: str) -> str:
    """Return the name of the 'main' metric used for averaging."""
    mtype = DATASET_METRIC_TYPE.get(dataset, "accuracy")
    if mtype in ("accuracy", "accuracy+qa_f1"):
        return "accuracy"
    elif mtype == "recall":
        return "recall"
    elif mtype == "rouge":
        return "rouge_l"
    elif mtype == "gec_f1":
        return "gec_f1"
    elif mtype == "model_metric":
        return "model_score"
    return "accuracy"


# ===========================================================================
# Main evaluation loop
# ===========================================================================

def main():
    args = parse_args()

    # Load files
    print(f"Loading predictions from:  {args.predictions}")
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loading benchmark from:    {args.benchmark}")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Build lookup: (category, dataset, item_id) → benchmark item
    benchmark_lookup: Dict[Tuple, dict] = {}
    for category, datasets in benchmark["data"].items():
        for dataset, ds_info in datasets.items():
            for item in ds_info["items"]:
                benchmark_lookup[(category, dataset, item["item_id"])] = item

    # Set up judge client if needed
    judge_client = None
    if args.judge_base_url and args.judge_api_key:
        from openai import OpenAI
        judge_client = OpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)
        print(f"Judge model: {args.judge_model} @ {args.judge_base_url}")
    else:
        print("[INFO] No --judge_base_url/--judge_api_key provided; "
              "ModelMetric datasets (geometry/prealgebra/precalculus) will be skipped.")

    # Per-dataset results
    dataset_results: Dict[str, Dict] = {}
    category_map: Dict[str, List[str]] = {}

    # Identify categories from predictions structure
    for category, datasets in predictions["data"].items():
        category_map[category] = list(datasets.keys())

    # Evaluate each dataset
    for category, datasets in predictions["data"].items():
        for dataset, ds_info in datasets.items():
            items = ds_info.get("items", [])
            print(f"\n[{category}] {dataset}: {len(items)} items")

            item_scores: List[Dict] = []
            for pred_item in items:
                item_id = pred_item["item_id"]
                bench_item = benchmark_lookup.get((category, dataset, item_id), {})

                # Merge: take ground_truth and question from benchmark
                merged = dict(pred_item)
                merged["ground_truth"] = bench_item.get("ground_truth", pred_item.get("ground_truth", ""))
                merged["question"] = bench_item.get("question", pred_item.get("question", ""))

                scores = eval_item(dataset, merged,
                                   judge_client=judge_client,
                                   judge_model=args.judge_model)
                item_scores.append({"item_id": item_id, **scores})

            # Aggregate
            metric_names = set()
            for s in item_scores:
                metric_names.update(k for k in s if k != "item_id")

            agg: Dict[str, float] = {}
            for mname in metric_names:
                vals = [s[mname] for s in item_scores if s.get(mname) is not None]
                agg[mname] = mean(vals) if vals else 0.0

            dataset_results[dataset] = {
                "category": category,
                "num_items": len(items),
                "per_item_scores": item_scores,
                "aggregate": agg,
            }

            # Print aggregate
            agg_str = "  ".join(f"{k}={v:.4f}" for k, v in agg.items())
            print(f"  → {agg_str}")

    # Category averages (macro over primary metric of each dataset)
    print("\n\n=== Category Averages ===")
    category_averages: Dict[str, float] = {}
    for category, ds_list in category_map.items():
        scores = []
        for dataset in ds_list:
            if dataset in dataset_results:
                pname = primary_metric_name(dataset)
                val = dataset_results[dataset]["aggregate"].get(pname)
                if val is not None:
                    scores.append(val)
        cat_avg = mean(scores)
        category_averages[category] = cat_avg
        print(f"  {category:<30}: {cat_avg:.4f}  (over {len(scores)} datasets)")

    # Global average (macro over all datasets)
    all_primary = []
    for dataset, res in dataset_results.items():
        pname = primary_metric_name(dataset)
        val = res["aggregate"].get(pname)
        if val is not None:
            all_primary.append(val)
    global_avg = mean(all_primary)
    print(f"\n  {'Global Average':<30}: {global_avg:.4f}  (over {len(all_primary)} datasets)")

    # Write results JSON
    output = {
        "inference_metadata": predictions.get("inference_metadata", {}),
        "global_average": global_avg,
        "category_averages": category_averages,
        "dataset_results": {
            ds: {
                "category": res["category"],
                "num_items": res["num_items"],
                "aggregate": res["aggregate"],
            }
            for ds, res in dataset_results.items()
        },
        "detailed_scores": {
            ds: res["per_item_scores"]
            for ds, res in dataset_results.items()
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nEvaluation results written to: {args.output}")


if __name__ == "__main__":
    main()
