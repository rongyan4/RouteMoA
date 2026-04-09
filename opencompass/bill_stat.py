import os
import json
import pandas as pd

# Configuration


root = '/home/1002/wangjize/wuhan/RouteMoA/opencompass/outputs/acl_rebuttal/20260217_192535/predictions/stop_0_90'

import os.path as osp  # ensure osp is available
import os  # ensure os is imported
model_name = os.path.basename(root.rstrip(os.sep))

datasets = ['math', 'gsm8k', 'sanitized_mbpp', 'ARC', 'lukaemon_mmlu', 'race-high', 'IFEval', 'bbh', 'GPQA_diamond', 'agieval', 'aime2024','openai_humaneval', 'alpaca_eval']  



# Initialize statistics tables (single model)
cost_data = {model_name: {dataset: None for dataset in datasets}}
latency_data = {model_name: {dataset: None for dataset in datasets}}
avg_cost_data = {model_name: {dataset: None for dataset in datasets}}

# Process only a single model path
model = model_name

print("model_name:", model_name)

model_path = root
if not os.path.exists(model_path):
    print("-----------YOU MAY HAVE MADE A MISTAKE ---------")
    print(f"Warning: model path {model_path} does not exist.")
else:

    for dataset in datasets:
        total_cost = 0.0
        total_latency = 0.0
        count = 0

        # Iterate over all files whose names start with the dataset name
        for file_name in os.listdir(model_path):
            if file_name.startswith(dataset):
                file_path = os.path.join(model_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.values():
                        total_cost += item.get('bill', {}).get('cost', 0)
                        total_latency += item.get('latency', 0)
                        count += 1

        if count > 0:
            cost_data[model][dataset] = round(total_cost, 2)
            latency_data[model][dataset] = round(total_latency / count, 2)
            # Average cost is scaled to per 10,000 questions
            avg_cost_data[model][dataset] = round(total_cost / count * 10000, 2)

# Convert to DataFrame
cost_df = pd.DataFrame.from_dict(cost_data, orient='index')
cost_df.reset_index(inplace=True)
cost_df.rename(columns={'index': 'model'}, inplace=True)

latency_df = pd.DataFrame.from_dict(latency_data, orient='index')
latency_df.reset_index(inplace=True)
latency_df.rename(columns={'index': 'model'}, inplace=True)

# Save as CSV
cost_file_name = 'emoa-math-agg-minstral_cost.csv'
latency_file_name = 'emoa-math-agg-minstral_latency.csv'
cost_df.to_csv(cost_file_name, index=False)
latency_df.to_csv(latency_file_name, index=False)

# Average cost statistics
avg_cost_df = pd.DataFrame.from_dict(avg_cost_data, orient='index')
avg_cost_df.reset_index(inplace=True)
avg_cost_df.rename(columns={'index': 'model'}, inplace=True)

avg_cost_file_name = 'emoa-math-agg-minstral_avg_cost.csv'
avg_cost_df.to_csv(avg_cost_file_name, index=False)

# Print results to the command line
print("\n=== Total Cost ===")
print(cost_df.to_string(index=False))
print("\n=== Average Cost per 10k questions ===")
print(avg_cost_df.to_string(index=False))
print("\n=== Average Latency ===")
print(latency_df.to_string(index=False))

print(f"Statistics complete, results saved to {cost_file_name}、{avg_cost_file_name} and {latency_file_name}")
