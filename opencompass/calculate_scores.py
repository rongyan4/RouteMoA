#----------------calculate scores distribution in the file-----------
# import json
# from collections import Counter

# # File paths
# file_path = "/cpfs01/user/chenyicheng/wangjize/opencompass/subset/race-high-test.json"

# # Initialize counter for max label values
# max_score_counts = Counter()

# # Read JSON file
# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Iterate through each data entry to extract max label value
# for item in data:
#     labels = item.get("labels", {})
#     if labels:  # Ignore empty label items
#         max_score = max(labels.values())
#         max_score_counts[max_score] += 1

# # Output distribution statistics
# print("Distribution of valid max label values:")
# for score, count in sorted(max_score_counts.items(), reverse=True):
#     print(f"{score}: {count}")



#--------------------determine -------------
# import json
# import os
# import re

# # -- Configuration paths -- #
# PREDICTIONS_DIR = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# RESULT_FILE_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/gsm8k.json"

# # -- Extract the last number from gold answers -- #
# def extract_final_number(text):
#     if not text:
#         return None
#     matches = re.findall(r'\d+(?:,\d+)?(?:\.\d+)?', text)
#     return matches[-1].replace(',', '') if matches else None

# # -- Read all gsm8k_i.json files in prediction folder -- #
# def load_all_predictions(pred_dir):
#     predictions = []
#     for fname in sorted(os.listdir(pred_dir)):
#         if fname.startswith("gsm8k_") and fname.endswith(".json"):
#             path = os.path.join(pred_dir, fname)
#             with open(path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 predictions.extend(list(data.values()))
#     return predictions

# # -- Load predictions and ground truth -- #
# def load_result_file(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         result_data = json.load(f)
#     return result_data["details"]

# # -- Main process -- #
# def main():
#     pred_items = load_all_predictions(PREDICTIONS_DIR)
#     results = load_result_file(RESULT_FILE_PATH)

#     assert len(pred_items) == len(results), f"Quantity mismatch: pred={len(pred_items)}, results={len(results)}"

#     total = len(pred_items)
#     correct = 0

#     for i, (pred_item, result_item) in enumerate(zip(pred_items, results)):
#         pred_gold = extract_final_number(pred_item.get("gold", ""))
#         pred_answer = result_item["pred"][0].replace(",", "")

#         if pred_gold == pred_answer:
#             correct += 1

#     acc = 100 * correct / total
#     print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%")

# if __name__ == "__main__":
#     main()



#--------------------------------Find how many are identical---------------

# import json
# import os
# import re

# # -- Configuration paths -- #
# PREDICTIONS_DIR = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# RESULT_FILE_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/gsm8k.json"

# # -- Extract the last number from gold -- #
# def extract_final_number(text):
#     if not text:
#         return None
#     matches = re.findall(r'\d+(?:,\d+)?(?:\.\d+)?', text)
#     return matches[-1].replace(',', '') if matches else None

# # -- Read prediction directory and merge all gsm8k_i.json files -- #
# def load_all_predictions(pred_dir):
#     predictions = []
#     for fname in sorted(os.listdir(pred_dir)):
#         if fname.startswith("gsm8k_") and fname.endswith(".json"):
#             path = os.path.join(pred_dir, fname)
#             with open(path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 predictions.extend(list(data.values()))
#     return predictions

# # -- Read result file -- #
# def load_result_file(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         result_data = json.load(f)
#     return result_data["details"]

# # -- Main function -- #
# def main():
#     pred_items = load_all_predictions(PREDICTIONS_DIR)
#     results = load_result_file(RESULT_FILE_PATH)

#     all_result_preds = [r["answer"][0].replace(",", "") for r in results]

#     total = len(pred_items)
#     correct = 0
#     not_found = 0
#     multiple_found = 0

#     for i, pred_item in enumerate(pred_items):
#         gold_ans = extract_final_number(pred_item.get("gold", ""))
#         if gold_ans is None:
#             print(f"[WARNING] No gold answer extracted for index {i}")
#             continue

#         match_count = sum(1 for p in all_result_preds if p == gold_ans)

#         if match_count == 1:
#             correct += 1
#         elif match_count == 0:
#             not_found += 1
#         else:
#             multiple_found += 1

#     print("\n====== Detailed Matching Statistics ======")
#     print(f"Total examples         : {total}")
#     print(f"✔️  Correct match (only one found) : {correct}")
#     print(f"❌  No match found               : {not_found}")
#     print(f"❌  Multiple matches found       : {multiple_found}")
#     accuracy = 100 * correct / total
#     print(f"\n✅ Accuracy (unique match only) : {accuracy:.2f}%")
#     print("==========================================")

# if __name__ == "__main__":
#     main()

#-----------------============

# import json
# import os
# import re
# from glob import glob
# from collections import Counter
# from tqdm import tqdm

# # ──── CONFIG ────────────────────────────────────────────────────────────────
# PRED_DIR = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# RESULT_FILE = ("/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/"
#                "20250429_093338/results/emoa-oracle/gsm8k.json")
# # ────────────────────────────────────────────────────────────────────────────


# def extract_answer(text: str) -> str | None:
#     """Return the final numeric answer as a *string* (commas stripped)."""
#     # 1) First look for \boxed{...}
#     m = re.search(r'\\boxed\{\s*([^\}]+?)\s*\}', text)
#     if m:
#         return m.group(1).replace(",", "").strip()

#     # 2) Otherwise take the last number appearing in the text
#     nums = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
#     if nums:
#         return nums[-1].replace(",", "").strip()

#     return None  # Extraction failed


# def load_pred_numbers(pred_dir: str) -> list[str | None]:
#     """Read gsm8k_*.json files and return extracted answers in file order."""
#     files = sorted(
#         glob(os.path.join(pred_dir, "gsm8k_*.json")),
#         key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
#     )

#     extracted: list[str | None] = []
#     for fp in tqdm(files, desc="Loading prediction shards"):
#         with open(fp, "r", encoding="utf-8") as f:
#             shard = json.load(f)
#         for obj in shard.values():
#             extracted.append(extract_answer(obj["prediction"]))
#     return extracted


# def load_result_predictions(result_path: str) -> list[str]:
#     """Return the list of model predictions stored in gsm8k.json."""
#     with open(result_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return [d["pred"][0] for d in data["details"]]


# def main() -> None:
#     preds = load_pred_numbers(PRED_DIR)
#     result_preds = load_result_predictions(RESULT_FILE)
#     result_counter = Counter(result_preds)

#     total = len(preds)
#     extracted_ok = sum(p is not None for p in preds)

#     correct = 0
#     print("\nExtracted answers (index ➟ value):")
#     for idx, ans in enumerate(preds):
#         print(f"{idx:5d} ➟ {ans}")
#         if ans is None:
#             continue
#         # "Correct only if exactly one matches"
#         if result_counter[ans] == 1:
#             correct += 1

#     accuracy = correct / total * 100
#     extract_rate = extracted_ok / total * 100

#     print("\n────────── Summary ──────────")
#     print(f"Total items                 : {total}")
#     print(f"Successfully extracted nums : {extracted_ok}  ({extract_rate:.2f}%)")
#     print(f"Correct (unique‑match rule) : {correct}")
#     print(f"Accuracy                    : {accuracy:.2f}%")
#     print("─────────────────────────────")


# if __name__ == "__main__":
#     main()


#---------------Find matching relationship between indices (mbpp)--------------
# import json
# import os

# # -- Path Configuration -- #
# FILE1_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/sanitized_mbpp.json"
# FILE2_DIR  = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"

# # Read file 1
# with open(FILE1_PATH, "r", encoding="utf-8") as f1:
#     data1 = json.load(f1)
# details = data1["details"]

# # List and sort all matching json files under file 2
# file2_files = sorted(
#     [fn for fn in os.listdir(FILE2_DIR) 
#          if fn.startswith("sanitized_mbpp_") and fn.endswith(".json")],
#     key=lambda fn: int(fn.split("_")[-1].split(".")[0])
# )

# print("file1_index\tfile2_index\tfile2_filename")

# # For each entry in file 1
# for item in details:
#     # Parse index of file 1 from example_abbr
#     abbr = item.get("example_abbr", "")
#     try:
#         file1_idx = int(abbr.split("_")[-1])
#     except ValueError:
#         # Skip if parsing fails
#         continue

#     # Concatenate answer array into a single string (preserve newlines)
#     answer_str = "\n".join(item.get("answer", [])).strip()

#     # Look for matches in all files 2
#     matched = False
#     for fn in file2_files:
#         path2 = os.path.join(FILE2_DIR, fn)
#         with open(path2, "r", encoding="utf-8") as f2:
#             data2 = json.load(f2)

#         # data2 is a dict like { "0": {...}, "1": {...}, ... }
#         for file2_idx_str, rec in data2.items():
#             gold_str = rec.get("gold", "").strip()
#             if gold_str == answer_str:
#                 # Found a match, print and break out of double loop
#                 print(f"{file1_idx}\t\t{file2_idx_str}\t\t{fn}")
#                 matched = True
#                 break
#         if matched:
#             break


#------------------Find matching relationship between indices (arc-c)-----------------------

# import json
# import os
# import re

# # -- Path Configuration -- #
# FILE1_PATH   = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/ARC-c.json"
# FILE2_DIR    = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# OUTPUT_PATH  = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/ARC-c_matches.json"

# # Only match ARC-c_<number>.json
# pattern = re.compile(r"^ARC-c_(\d+)\.json$")

# # Collect and sort all files 2 matching the format
# file2_list = []
# for fn in os.listdir(FILE2_DIR):
#     m = pattern.match(fn)
#     if not m:
#         continue
#     idx = int(m.group(1))
#     file2_list.append((idx, fn))

# file2_list.sort(key=lambda x: x[0])
# file2_files = [fn for _, fn in file2_list]

# # Read file 1
# with open(FILE1_PATH, "r", encoding="utf-8") as f1:
#     data1 = json.load(f1)

# details = data1.get("details", {})

# # Used to store all match results
# matches = []

# # Iterate through each record in file 1
# for file1_idx_str, rec1 in details.items():
#     origin_pred = rec1.get("origin_prediction", "").strip()
#     found = False

#     # Look for matches in the file 2 list
#     for fn in file2_files:
#         path2 = os.path.join(FILE2_DIR, fn)
#         with open(path2, "r", encoding="utf-8") as f2:
#             data2 = json.load(f2)

#         # data2 is a dictionary like {"0": {...}, "1": {...}, ...}
#         for file2_idx_str, rec2 in data2.items():
#             pred2 = rec2.get("prediction", "").strip()
#             if pred2 == origin_pred:
#                 # Match successful, record in list and break
#                 matches.append({
#                     "file1_index":    file1_idx_str,
#                     "file2_index":    file2_idx_str,
#                     "file2_filename": fn
#                 })
#                 found = True
#                 break

#         if found:
#             break

#     # If you want to record unmatched items, uncomment the line below:
#     # if not found:
#     #     matches.append({"file1_index": file1_idx_str, "file2_index": None, "file2_filename": None})

# # Write match results to JSON file
# with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
#     json.dump(matches, fout, indent=2, ensure_ascii=False)

# print(f"Done! {len(matches)} matches written to {OUTPUT_PATH}")

 
 #########################gsm8k##############
# import json
# import os
# import re

# # -- Configuration section -- #
# FILE3_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/subset/gsm8k-test.json"
# PRED_DIR   = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# FILE2_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/gsm8k.json"
# # -- Configuration end -- #

# def load_subset(path):
#     """Read file 3, return prompt list and max label value list for each entry"""
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     prompts, max_labels = [], []
#     for entry in data:
#         p = entry['prompt'][0]['prompt'].strip()
#         prompts.append(p)
#         max_labels.append(max(entry['labels'].values()))
#     return prompts, max_labels


# def load_predictions_from_dir(dirpath):
#     """
#     Scan all gsm8k_*.json files in the directory, read in numerical order and concatenate origin_prompt from each file.
#     Return a long list, ordered by ascending keys "0","1",... within files, and ascending file names.
#     """
#     # Find all files matching gsm8k_*.json
#     files = [
#         fn for fn in os.listdir(dirpath)
#         if re.match(r"gsm8k_(\d+)\.json$", fn)
#     ]
#     # Sort numerically ascending
#     files.sort(key=lambda fn: int(re.match(r"gsm8k_(\d+)\.json$", fn).group(1)))

#     all_prompts = []
#     for fn in files:
#         full_path = os.path.join(dirpath, fn)
#         with open(full_path, 'r', encoding='utf-8') as f:
#             part = json.load(f)
#         # part is a dict with keys "0","1",...
#         # Extract origin_prompt ascending by key
#         for idx in sorted(part.keys(), key=lambda k: int(k)):
#             p = part[idx]['origin_prompt'][0]['prompt'].strip()
#             all_prompts.append(p)
#     return all_prompts

# def load_results(path):
#     """Read file 2, return details list"""
#     with open(path, 'r', encoding='utf-8') as f:
#         results = json.load(f)
#     return results['details']

# def compute_score(max_label, correct_flag):
#     """
#     Calculate by formula: result = max_label + (1 - max_label) * int(correct_flag)
#     """
#     return max_label + (1 - max_label) * int(bool(correct_flag))

# def main():
#     # 1. Load data
#     subset_prompts, subset_max_labels = load_subset(FILE3_PATH)
#     pred_prompts        = load_predictions_from_dir(PRED_DIR)
#     details             = load_results(FILE2_PATH)

#     total = len(subset_prompts)
#     # 2. Match
#     matched_indices = [i for i, p in enumerate(subset_prompts) if p in pred_prompts]
#     matched = len(matched_indices)

#     print(f"Total prompts in subset: {total}")
#     print(f"Number of prompts matched in predictions: {matched}")

#     # 3. Calculate and output item by item
#     scores = []
#     for i in matched_indices:
#         max_label    = subset_max_labels[i]
#         correct_flag = details[i]['correct'][0]
#         sc = compute_score(max_label, correct_flag)
#         scores.append(sc)
#         print(f"  [{i}] max_label={max_label:.3f}, correct={correct_flag}, score={sc:.3f}")

#     total_score = sum(scores)/matched
#     print(f"\nFinal total score: {total_score:.3f}")
#     print(f"Total prompts in subset: {total}")
#     print(f"Number of prompts matched in predictions: {matched}")

# if __name__ == "__main__":
#     main()

#----------------------------------
#-----------------math-------------
#!/usr/bin/env python3
# compute_score.py

# import json
# import os
# import re

# # -- Configuration section -- #
# # Subset file (file 3)
# FILE3_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/subset/math-test.json"
# # Prediction result directory (file 1 concatenated from math_0.json, math_1.json ...)
# PRED_DIR   = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# # Summary result file (file 2)
# FILE2_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/math.json"
# # -- Configuration end -- #

# def load_subset(path):
#     """Read subset file, return prompt list and max label value list for each entry"""
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     prompts, max_labels = [], []
#     for entry in data:
#         p = entry['prompt'][0]['prompt'].strip()
#         prompts.append(p)
#         max_labels.append(max(entry['labels'].values()))
#     return prompts, max_labels

# def load_predictions_from_dir(dirpath):
#     """
#     Scan all math_*.json files in directory, sort by numerical suffix,
#     and concatenate origin_prompt from each file into a large list.
#     """
#     pattern = re.compile(r"math_(\d+)\.json$")
#     files = [fn for fn in os.listdir(dirpath) if pattern.match(fn)]
#     files.sort(key=lambda fn: int(pattern.match(fn).group(1)))

#     all_prompts = []
#     for fn in files:
#         full_path = os.path.join(dirpath, fn)
#         with open(full_path, 'r', encoding='utf-8') as f:
#             part = json.load(f)
#         # part is a dict with keys "0","1",... Extract origin_prompt in ascending key order
#         for idx in sorted(part.keys(), key=lambda k: int(k)):
#             p = part[idx]['origin_prompt'][0]['prompt'].strip()
#             all_prompts.append(p)
#     return all_prompts

# def load_results(path):
#     """Read summary result file, return details list"""
#     with open(path, 'r', encoding='utf-8') as f:
#         results = json.load(f)
#     return results['details']

# def compute_score(max_label, correct_flag):
#     """
#     Calculate by formula: result = max_label + (1 - max_label) * int(correct_flag)
#     """
#     return max_label + (1 - max_label) * int(bool(correct_flag))

# def main():
#     # 1. Load data
#     subset_prompts, subset_max_labels = load_subset(FILE3_PATH)
#     pred_prompts        = load_predictions_from_dir(PRED_DIR)
#     details             = load_results(FILE2_PATH)

#     total = len(subset_prompts)
#     # 2. Match subset prompts positions in the prediction list
#     matched_indices = [i for i, p in enumerate(subset_prompts) if p in pred_prompts]
#     matched = len(matched_indices)

    

#     # 3. Calculate scores item by item and accumulate
#     scores = []
#     for i in matched_indices:
#         max_label    = subset_max_labels[i]
#         correct_flag = details[i]['correct'][0]  # True/False
#         sc = compute_score(max_label, correct_flag)
#         scores.append(sc)
#         print(f"[{i}] max_label={max_label:.3f}, correct={correct_flag}, score={sc:.3f}")

#     total_score = sum(scores)/matched
#     print(f"\nFinal total score: {total_score:.3f}")
#     print(f"Total prompts in subset: {total}")
#     print(f"Number of prompts matched in predictions: {matched}")

# if __name__ == "__main__":
#     main()


#--------------------------mbpp-------------------



# import json
# import os
# import re
# from pathlib import Path
# from typing import Dict, List, Tuple

# # ── CONFIG ──────────────────────────────────────────────────────────────────── #
# FILE3_PATH = Path("/cpfs01/user/chenyicheng/wangjize/opencompass/subset/mbpp-test.json")
# PRED_DIR   = Path("/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle")
# FILE2_PATH = Path("/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/sanitized_mbpp.json")
# # ───────────────────────────────────────────────────────────────────────────── #

# # ╭──────────────────────────── Helper functions ─────────────────────────────╮ #
# def norm(text: str) -> str:
#     """Trim both ends and normalise internal whitespace/new‑lines."""
#     return "\n".join(line.rstrip() for line in text.strip().splitlines())

# def load_subset(path: Path) -> Tuple[List[str], List[float]]:
#     """Return (prompts, max_labels) from mbpp‑test.json."""
#     with path.open("r", encoding="utf-8") as f:
#         data = json.load(f)
#     prompts, max_labels = [], []
#     for entry in data:
#         prompts.append(norm(entry["prompt"][0]["prompt"]))
#         max_labels.append(max(entry["labels"].values()))
#     return prompts, max_labels


# def load_predictions_from_dir(dir_path: Path) -> Dict[str, Dict[str, str]]:
#     """
#     Scan sanitized_mbpp_*.json files and build a dict:
#         prompt_text → {"gold": gold_str}
#     We use the *last* HUMAN message inside each origin_prompt sequence.
#     """
#     pattern = re.compile(r"sanitized_mbpp_(\d+)\.json$")
#     files = sorted(
#         (f for f in dir_path.iterdir() if pattern.match(f.name)),
#         key=lambda p: int(pattern.match(p.name).group(1))
#     )

#     pred_map: Dict[str, Dict[str, str]] = {}
#     for fp in files:
#         with fp.open("r", encoding="utf-8") as f:
#             part = json.load(f)
#         for idx_key in sorted(part.keys(), key=int):
#             seq = part[idx_key]["origin_prompt"]
#             # last HUMAN prompt
#             human_prompts = [item["prompt"] for item in seq if item.get("role") == "HUMAN"]
#             if not human_prompts:
#                 raise ValueError(f"{fp.name}[{idx_key}] contains no HUMAN prompt")
#             prompt_text = norm(human_prompts[-1])
#             gold_text   = norm(part[idx_key]["gold"])
#             pred_map[prompt_text] = {"gold": gold_text}
#     return pred_map


# def load_answer_map(path: Path) -> Dict[str, bool]:
#     """
#     Build {answer_str → is_correct_bool} using FILE‑2 details.
#     """
#     with path.open("r", encoding="utf-8") as f:
#         results = json.load(f)
#     answer_map: Dict[str, bool] = {}
#     for detail in results["details"]:
#         answer_str = norm(detail["answer"][0])
#         is_correct = bool(detail["correct"][0]["is_correct"])
#         answer_map[answer_str] = is_correct
#     return answer_map


# def compute_score(max_label: float, correct_flag: bool) -> float:
#     """Custom score rule."""
#     return max_label + (1 - max_label) * int(correct_flag)


# def main() -> None:
#     # 1. Load all inputs
#     subset_prompts, subset_max_labels = load_subset(FILE3_PATH)
#     pred_map        = load_predictions_from_dir(PRED_DIR)
#     answer_map      = load_answer_map(FILE2_PATH)

#     total   = len(subset_prompts)
#     matched = 0
#     scores  = []

#     for prompt_text, max_label in zip(subset_prompts, subset_max_labels):
#         pred_entry = pred_map.get(prompt_text)
#         if pred_entry is None:
#             continue  # this subset prompt never appeared in predictions
#         matched += 1
#         gold_text   = pred_entry["gold"]
#         correct_flag = answer_map.get(gold_text, False)  # default False if not found
#         score = compute_score(max_label, correct_flag)
#         scores.append(score)
#         print(f"[{matched:03d}] max_label={max_label:.3f}  "
#               f"correct={correct_flag}  score={score:.3f}")

#     # 2. Summary
#     avg_score = sum(scores) / len(scores) if scores else 0.0
#     print("\n───────────── Summary ─────────────")
#     print(f"Total prompts in subset : {total}")
#     print(f"Matched prompts         : {matched}")
#     print(f"Average score           : {avg_score:.6f}")
#     print("───────────────────────────────────")


# if __name__ == "__main__":
#     main()


###############race-high###############
#!/usr/bin/env python3
# compute_score.py

# import json
# import os
# import re

# # -- Configuration section -- #
# FILE3_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/subset/race-high-test.json"
# PRED_DIR   = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/predictions/emoa-oracle"
# FILE2_PATH = "/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/moa_0429/20250429_093338/results/emoa-oracle/race-high.json"
# # -- Configuration end -- #


# def load_subset(path):
#     """
#     Read race-high-test.json, return:
#       - prompts: list of prompt texts in the subset
#       - max_labels: list of max label values per entry (race-high has no labels, uniformly set to 1.0)
#     """
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     prompts, max_labels = [], []
#     for entry in data:
#         p = entry['prompt'][0]['prompt'].strip()
#         prompts.append(p)
#         max_labels.append(max(entry['labels'].values()))

#     return prompts, max_labels


# def load_predictions_from_dir(dirpath):
#     """
#     Scan all race-high_*.json files in directory, sort by numerical suffix,
#     and extract origin_prompt[0] of each entry in index order.
#     Return a large list, aligned with subset prompts.
#     """
#     pattern = re.compile(r"race-high_(\d+)\.json$")
#     files = sorted(
#         [fn for fn in os.listdir(dirpath) if pattern.match(fn)],
#         key=lambda fn: int(pattern.match(fn).group(1))
#     )
#     all_prompts = []
#     for fn in files:
#         full_path = os.path.join(dirpath, fn)
#         with open(full_path, 'r', encoding='utf-8') as f:
#             part = json.load(f)
#         # part is dict, key is numeric string
#         for idx in sorted(part.keys(), key=lambda k: int(k)):
#             entry = part[idx]
#             p = entry['origin_prompt'][0]['prompt'].strip()
#             all_prompts.append(p)
#     return all_prompts


# def load_results(path):
#     """
#     Read race-high.json, return details dict
#     """
#     with open(path, 'r', encoding='utf-8') as f:
#         results = json.load(f)
#     return results['details']  # dict: index_str -> record


# def compute_score(max_label, correct_flag):
#     """
#     Calculate by formula: result = max_label + (1 - max_label) * int(correct_flag)
#     """
#     return max_label + (1 - max_label) * int(bool(correct_flag))


# def main():
#     subset_prompts, subset_max_labels = load_subset(FILE3_PATH)
#     pred_prompts        = load_predictions_from_dir(PRED_DIR)
#     details             = load_results(FILE2_PATH)

#     total = len(subset_prompts)
#     matched_indices = [i for i, p in enumerate(subset_prompts) if p in pred_prompts]
#     matched = len(matched_indices)

#     print(f"Total prompts: {total}")
#     print(f"Matched prompts: {matched}")

#     scores = []
#     for i in matched_indices:
#         max_label = subset_max_labels[i]
#         record = details.get(str(i), {})
#         # File 2 correct judgment: predictions == references
#         correct_flag = (record.get('predictions') == record.get('references'))
#         score = compute_score(max_label, correct_flag)
#         scores.append(score)
#         print(max_label)
#         print(f"[{i}] correct={correct_flag}, score={score:.3f}")

#     avg_score = sum(scores) / len(scores) if scores else 0
#     print(f"\nFinal average score: {avg_score:.3f}")

# if __name__ == "__main__":
#     main()



#------------mmlu----------

# import json
# import os
# import re

# # Root directory Configuration
# BASE_DIR = "/cpfs01/user/chenyicheng/wangjize/opencompass"
# PRED_BASE_DIR = os.path.join(BASE_DIR, "outputs/moa_0429/20250429_093338/predictions/emoa-oracle")
# RES_BASE_DIR  = os.path.join(BASE_DIR, "outputs/moa_0429/20250429_093338/results/emoa-oracle")
# SUBSET_DIR    = os.path.join(BASE_DIR, "subset")

# # List of tasks to process, each containing subset suffix
# TASKS = [
#     "anatomy",
#     "clinical_knowledge",
#     "college_biology",
#     "college_medicine",
#     "medical_genetics",
# ]


# def load_subset(path):
#     """Read subset file, return prompt list and max label value list for each entry"""
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     prompts, max_labels = [], []
#     for entry in data:
#         p = entry['prompt'][0]['prompt'].strip()
#         prompts.append(p)
#         max_labels.append(max(entry['labels'].values()))
#     return prompts, max_labels

# def load_predictions(pred_dir, prefix):
#     """
#     Scan all files named prefix_number.json under pred_dir, ascending numerically,
#     extract origin_prompt[0] of each entry, return large list.
#     """
#     pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.json$")
#     files = sorted(
#         [fn for fn in os.listdir(pred_dir) if pattern.match(fn)],
#         key=lambda fn: int(pattern.match(fn).group(1)))

#     all_prompts = []
#     for fn in files:
#         path = os.path.join(pred_dir, fn)
#         part = json.load(open(path, 'r', encoding='utf-8'))
#         for idx in sorted(part.keys(), key=lambda k: int(k)):
#             prompt = part[idx]['origin_prompt'][0]['prompt'].strip()
#             all_prompts.append(prompt)
#     return all_prompts


# def load_results(results_path):
#     """
#     Load results.json, return details dict.
#     """
#     with open(results_path, 'r', encoding='utf-8') as f:
#         results = json.load(f)
#     return results['details']


# def compute_score(max_label, correct_flag):
#     """
#     Calculate by formula: result = max_label + (1 - max_label) * int(correct_flag)
#     """
#     return max_label + (1 - max_label) * int(bool(correct_flag))

# def main():
#     summary = []  # Store (task, avg_score, sample_count)

#     for task in TASKS:
#         # Construct paths
#         subset_file = os.path.join(SUBSET_DIR, f"mmlu-test_{task}.json")
#         pred_prefix = f"lukaemon_mmlu_{task}"
#         results_file = os.path.join(RES_BASE_DIR, f"lukaemon_mmlu_{task}.json")

#         # Load data
#         subset_prompts, max_labels = load_subset(subset_file)
#         pred_prompts = load_predictions(PRED_BASE_DIR, pred_prefix)
#         details = load_results(results_file)

#         # Match and score
#         matched_indices = [i for i, p in enumerate(subset_prompts) if p in pred_prompts]
#         scores = []
#         for i in matched_indices:
#             rec = details.get(str(i), {})
#             correct_flag = rec.get('predictions', []) == rec.get('references', [])
#             scores.append(compute_score(max_labels[i], correct_flag))
#             print(max_labels[i], correct_flag)

#         count = len(scores)
#         avg = sum(scores) / count if count else 0
#         summary.append((task, avg, count))
#         print(f"Task {task}: samples={count}, avg_score={avg:.3f}")

#     # Calculate weighted average
#     total_samples = sum(cnt for _, _, cnt in summary)
#     weighted_avg = sum(avg * cnt for _, avg, cnt in summary) / total_samples if total_samples else 0
#     print(f"\nOverall weighted average score: {weighted_avg:.3f} (total_samples={total_samples})")

# if __name__ == '__main__':
#     main()




#-------------------arc-c----------

import json
import os
import re

# -- Path Configuration -- #
BASE_DIR    = "/cpfs01/user/chenyicheng/wangjize/opencompass"
PRED_DIR    = os.path.join(BASE_DIR, "outputs/moa_0429/20250429_093338/predictions/emoa-oracle")
RESULT_FILE = os.path.join(BASE_DIR, "outputs/moa_0429/20250429_093338/results/emoa-oracle/ARC-c.json")
SUBSET_FILE = os.path.join(BASE_DIR, "subset/arc-c-test.json")

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_subset_prompts(path):
    """Read subset file, return prompt list and max label value list for each entry"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts, max_labels = [], []
    for entry in data:
        p = entry['prompt'][0]['prompt'].strip()
        prompts.append(p)
        max_labels.append(max(entry['labels'].values()))
    return prompts, max_labels

def load_pred_prompts(prefix="ARC-c"):
    """
    Scan all ARC-c_<n>.json in PRED_DIR, concatenate origin_prompt[0] ascending by n.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.json$")
    files = [fn for fn in os.listdir(PRED_DIR) if pattern.match(fn)]
    files.sort(key=lambda fn: int(pattern.match(fn).group(1)))

    prompts = []
    for fn in files:
        part = load_json(os.path.join(PRED_DIR, fn))
        for idx in sorted(part.keys(), key=lambda k: int(k)):
            prompts.append(part[idx]['origin_prompt'][0]['prompt'].strip())
    return prompts

def load_details(path):
    data = load_json(path)
    return data['details']  # dict of index_str -> record

def compute_score(max_label, correct_flag):
    """
    Calculate by formula: result = max_label + (1 - max_label) * int(correct_flag)
    """
    return max_label + (1 - max_label) * int(bool(correct_flag))


def main():
    # 1. Load data
    subset_prompts, max_labels = load_subset_prompts(SUBSET_FILE)
    pred_prompts   = load_pred_prompts()
    details        = load_details(RESULT_FILE)

    # 2. Match and score
    matched = [i for i, p in enumerate(subset_prompts) if p in pred_prompts]
    scores = []
    for i in matched:
        rec = details.get(str(i), {})
        correct = rec.get('predictions') == rec.get('references')
        a = compute_score(max_labels[i], correct)
        scores.append(a)

    count = len(scores)
    avg   = sum(scores) / count if count else 0.0

    # 3. Output results
    print(f"ARC-c matched samples: {count}")
    print(f"ARC-c average accuracy: {avg:.3f}")

if __name__ == "__main__":
    main()
