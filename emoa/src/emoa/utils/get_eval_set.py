import json
import datasets


def get_eval_set(input_path: str) -> datasets.Dataset:
    if input_path == "alpaca":
        eval_set = datasets.load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
        )["eval"]
        eval_set = eval_set.remove_columns(["output", "generator"])
    elif input_path.endswith(".json"):
        data = []
        with open(input_path) as f:
            data = json.load(f)
        eval_set = datasets.Dataset.from_list(data)
    elif input_path.endswith(".jsonl"):
        data = []
        with open(input_path) as f:
            for line in f:
                data.append(json.loads(line))
        eval_set = datasets.Dataset.from_list(data)
    else:
        raise ValueError(f"Unknown type of input_path: {input_path}, should be apalca or json or jsonl")
    
    return eval_set
