# Benchmark Evaluation Data & Reproduction Scripts

This folder contains the benchmark data, model predictions, and evaluation scripts used in our paper. Everything here is self-contained and designed for independent reproduction — no access to the internal iEval platform is required.

---

## File Overview

```
open_source_data_paper/
├── benchmark_questions.json   # Questions, ground-truth answers, and prompts for all 30 datasets
├── moa.json                   # MoA model responses + per-item evaluation results
├── routemoa.json              # RouteMoA model responses + per-item evaluation results
├── smoa.json                  # SMoA model responses + per-item evaluation results
├── summary_all.json           # Cross-model comparison summary (per-dataset + category averages)
├── inference.py               # Inference script: run any model on the benchmark
└── evaluate.py                # Evaluation script: compute all metrics from predictions
```

---

## Benchmark Overview

The benchmark covers **30 datasets** across **5 categories**, with **15 items per dataset** (450 items total). Datasets and metrics are as follows:

| Category | Dataset | Metric |
|---|---|---|
| **Language Understanding** | lcqmc | Accuracy |
| | ocnli | Accuracy |
| | sst2 | Accuracy |
| | cola | Accuracy |
| | mrpc | Accuracy |
| | msra | Recall |
| | qqp | Accuracy |
| | sts_b | Accuracy |
| | ag_news | Accuracy |
| | qnli | Accuracy |
| | chid_baidu | Accuracy |
| **Reading & QA** | webqa | Accuracy + QA-F1 |
| | c3 | Accuracy |
| | cmrc | Accuracy + QA-F1 |
| | race | Accuracy |
| | story_cloze_test | Accuracy |
| **Logic Reasoning** | cluewsc2020 | Accuracy |
| | winograd_wsc | Accuracy |
| | truthful_qa | Accuracy |
| **Math Reasoning** | bigmath | Accuracy (NumberParser) |
| | gsm8k_test_100 | Accuracy (NumberParser) |
| | GAOKAO-2023_Math_en | Accuracy (NumberParser) |
| | geometry | LLM Judge (gpt-4o, 0–4) |
| | prealgebra | LLM Judge (gpt-4o, 0–4) |
| | precalculus | LLM Judge (gpt-4o, 0–4) |
| **Language Generation** | word_manipulation_v2 | Accuracy |
| | nlpcc2017_task2 | Accuracy |
| | lcsts | RougeMetric (Rouge-L) |
| | nlpcc2018_task2 | GEC F1 |
| | conll2014 | GEC F1 |

---

## Data Format

### `benchmark_questions.json`

Contains questions and ground-truth answers for all 30 datasets, organized by category → dataset → items. Each item has:

```json
{
  "dataset": "lcqmc",
  "item_id": "8787",
  "question": "...",
  "ground_truth": "1",
  "full_prompt": "句子'...'和句子'...'是否有相同的语义?..."
}
```

Use `full_prompt` as the model input. `question` and `ground_truth` are used during evaluation.

### `moa.json` / `routemoa.json` / `smoa.json`

Same structure as `benchmark_questions.json`, with an additional `model_response` field per item containing the model's raw text output. These are the raw responses used in the paper.

### `summary_all.json`

Cross-model comparison with per-dataset results and aggregated averages. Key top-level fields:
- `per_dataset` — accuracy / latency / cost for each model on each dataset
- `category_avg` — macro average within each of the 5 categories
- `global_avg` — macro average equally over all 30 datasets
- `global_avg_by_category` — macro average first by category, then over 5 categories

---

## Paper Results

### Global Average (macro over 30 datasets)

| Model | Accuracy | Avg Latency (s) | Normalized Cost |
|---|---|---|---|
| MoA | 0.7151 | 243.7 | 436.9 |
| **RouteMoA** | **0.7835** | **86.9** | **44.2** |
| SMoA | 0.6943 | 188.4 | 97.1 |

### Per-Category Average

| Category | Model | Accuracy | Avg Latency (s) | Normalized Cost |
|---|---|---|---|---|
| **Language Understanding** | MoA | 0.8341 | 126.3 | 321.73 |
| | RouteMoA | 0.8402 | 43.4 | 24.90 |
| | SMoA | 0.7844 | 76.4 | 47.78 |
| **Reading & QA** | MoA | 0.8800 | 101.4 | 303.45 |
| | RouteMoA | 0.8800 | 27.8 | 14.21 |
| | SMoA | 0.8533 | 94.1 | 53.85 |
| **Logic Reasoning** | MoA | 0.9333 | 134.2 | 385.32 |
| | RouteMoA | 0.9556 | 58.9 | 28.73 |
| | SMoA | 0.9111 | 76.1 | 57.16 |
| **Math Reasoning** | MoA | 0.4972 | 619.5 | 751.08 |
| | RouteMoA | 0.7333 | 211.4 | 94.63 |
| | SMoA | 0.5306 | 471.2 | 232.54 |
| **Language Generation** | MoA | 0.4187 | 258.9 | 477.47 |
| | RouteMoA | 0.5190 | 109.3 | 65.68 |
| | SMoA | 0.4031 | 257.1 | 110.45 |

> **Latency** is the average per-item API response time in seconds.  
> **Normalized Cost** is a relative token-cost metric (proportional to input+output tokens weighted by model price).

---

## Reproducing the Results

### Step 1 — Install dependencies

```bash
pip install openai
pip install lawrouge          # for RougeMetric (lcsts)
pip install pkuseg nltk       # for GEC_f1 (nlpcc2018_task2, conll2014)
```

For GEC F1 (grammar error correction), `m2score` is also required:
```bash
git clone https://github.com/nusnlp/m2scorer
# Add the cloned directory to your PYTHONPATH:
export PYTHONPATH=/path/to/m2scorer:$PYTHONPATH
```

### Step 2 — Run inference

```bash
python inference.py \
    --base_url  https://api.openai.com/v1 \
    --api_key   YOUR_API_KEY \
    --model     gpt-4o \
    --input     benchmark_questions.json \
    --output    predictions.json \
    --workers   4
```

This writes `predictions.json` with the same structure as `benchmark_questions.json`, plus a `model_response` field for each item. If interrupted, re-running the command will **resume** from where it left off (already-completed items are skipped).

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--base_url` | *(required)* | OpenAI-compatible API base URL |
| `--api_key` | *(required)* | API key |
| `--model` | *(required)* | Model name (e.g. `gpt-4o`, `qwen-max`) |
| `--input` | `benchmark_questions.json` | Input benchmark file |
| `--output` | `predictions.json` | Output predictions file |
| `--workers` | `4` | Number of parallel API request threads |
| `--max_retries` | `3` | Retries per item on API error |

### Step 3 — Run evaluation

```bash
python evaluate.py \
    --predictions     predictions.json \
    --benchmark       benchmark_questions.json \
    --output          eval_results.json \
    --judge_base_url  https://api.openai.com/v1 \
    --judge_api_key   YOUR_OPENAI_KEY \
    --judge_model     gpt-4o
```

`--judge_base_url` and `--judge_api_key` are only required for the three math datasets that use an LLM judge (`geometry`, `prealgebra`, `precalculus`). All other datasets are evaluated locally without any API calls.

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--predictions` | `predictions.json` | Predictions file (output of inference.py) |
| `--benchmark` | `benchmark_questions.json` | Benchmark questions file |
| `--output` | `eval_results.json` | Where to write evaluation results |
| `--judge_base_url` | `None` | API base URL for LLM judge (ModelMetric datasets) |
| `--judge_api_key` | `None` | API key for LLM judge |
| `--judge_model` | `gpt-4o` | Judge model name |
| `--workers` | `4` | Parallel workers for judge API calls |

The script prints per-dataset scores, category averages, and a global average to stdout, and writes a full `eval_results.json` including per-item scores.

---

## Evaluation Metrics — Technical Details

| Metric | Datasets | Implementation |
|---|---|---|
| **Accuracy (keyword map)** | lcqmc, ocnli, sst2, cola, mrpc, qqp, sts_b, ag_news, qnli, race, story_cloze_test, cluewsc2020, truthful_qa | Map first keyword found in response to a canonical label, then compare with ground truth |
| **Accuracy (NumberParser)** | bigmath (last), gsm8k_test_100 (last 3 lines), GAOKAO-2023_Math_en (last 4 lines) | Extract the last number from the response (or the last N lines thereof) |
| **Accuracy (direct)** | chid_baidu, c3, word_manipulation_v2, nlpcc2017_task2, winograd_wsc | Direct string containment check between response and reference |
| **Recall** | msra | Split ground-truth NER entities by `[;,，；]`; compute fraction found in response |
| **QA-F1** | webqa, cmrc | Character-level token F1 (Chinese) between response and reference |
| **RougeMetric** | lcsts | `lawrouge` Rouge-1 (recall), Rouge-2 (recall), Rouge-L (F1) with `isChinese=True` |
| **GEC F1** | nlpcc2018_task2, conll2014 | Extract corrected sentence via regex, tokenize with pkuseg (CN) / nltk (EN), compute F1 via m2score levenshtein; 2-minute timeout per item |
| **ModelMetric** | geometry, prealgebra, precalculus | GPT-4o judge prompt (0–4 scale), parse `Score: N` from response, normalize to [0, 1] by dividing by 4 |

The **primary metric** used for category and global averages is: Accuracy for most datasets, Recall for msra, Rouge-L for lcsts, GEC F1 for grammar correction, and normalized model score for math judge datasets.
