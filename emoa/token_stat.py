import argparse
import tiktoken
from transformers import AutoTokenizer
from emoa.utils import inject_references_to_messages
from typing import Dict, List
import pandas as pd
import numpy as np
from textwrap import dedent
import copy
import json
from tqdm import tqdm


tokenizers = dict()

# INTERNAL_PROMPT = dedent(
# """
# You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

# Responses from models:
# """
# )

INTERNAL_PROMPT = dedent(
"""
You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Keep your answers to a moderate, shorter length, not too long, no longer than the longest reference answer unless it is very necessary, and make sure they are accurate and concise.

Responses from models:
"""
)

AGG_PROMPT = dedent(
"""
You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Keep your answers to a moderate, shorter length, not too long, no longer than the longest reference answer unless it is very necessary, and make sure they are accurate and concise.

Responses from models:
"""
)


def organize_input_output(line: Dict, internal_prompt: str, agg_prompt: str):
    models = [key for key, value in line['internal'][0].items()]
    input_msg, output_msg = dict(), dict()
    input_message_per_model = []
    for item in line['prompt']:
        input_message_per_model.append(item['content'])
    for i in range(len(line['internal'])):
        refs = line['internal'][i]
        references = [item for key, item in refs.items()]
        if line['prompt'][0]['role'] == 'system':
            orig_system_prompt = line['prompt'][0]['content']
        else:
            orig_system_prompt = None
        if i == len(line['internal']) - 1:
            system_prompt_to_inject = agg_prompt
        else:
            system_prompt_to_inject = internal_prompt
        concat_msg = inject_references_to_messages(
            messages=line['prompt'],
            references=references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=system_prompt_to_inject
        )
        for item in concat_msg:
            input_message_per_model.append(item['content'])

    for model in models:
        input_msg[model] = copy.deepcopy(input_message_per_model)
        output_msg[model] = []
    
    for refs in line['internal']:
        for model, ref in refs.items():
            output_msg[model].append(ref)
    
    output_msg[list(line['answer'].keys())[0]].append(list(line['answer'].values())[0])
    return input_msg, output_msg


def count_tokens(text: str, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def get_token_len(msg: List):
    token_len = {model: 0 for model in msg.keys()}
    for model, values in msg.items():
        for value in values:
            token_len[model] += count_tokens(value, tokenizers[model])
    return token_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='outputs/emoa_v1_1/mtbench.jsonl')
    parser.add_argument('-p', '--price_file', type=str, default='api_price.csv')
    parser.add_argument('-o', '--output_file', type=str, default='emoa_v1_1_mtbench_cost.csv')
    args = parser.parse_args()

    price = pd.read_csv(args.price_file, index_col=0)
    total_input_token_len, total_output_token_len, total_price_per_model = dict(), dict(), dict()
    models = []
    with open(args.file, 'r') as f:
        for line in f:
            line = json.loads(line)
            models = [key for key, value in line['internal'][0].items()]
            break
    for model in models:
        total_input_token_len[model] = 0
        total_output_token_len[model] = 0
        total_price_per_model[model] = 0
        tokenizers[model] = AutoTokenizer.from_pretrained(model)

    with open(args.file, 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            models = [key for key, value in line['internal'][0].items()]
            input_msg, output_msg = organize_input_output(line, INTERNAL_PROMPT, AGG_PROMPT)
            input_token_len = get_token_len(input_msg)
            output_token_len = get_token_len(output_msg)
            for model in models:
                total_input_token_len[model] += input_token_len[model]
                total_output_token_len[model] += output_token_len[model]
    
    total_price_moa = 0
    for model in models:
        input_len = total_input_token_len[model]
        input_price = price.loc['input_price/Mtokens', model]
        output_len = total_output_token_len[model]
        output_price = price.loc['output_price/Mtokens', model]
        total_price_per_model[model] = input_len * input_price / 1000000 \
                                     + output_len * output_price / 1000000
        total_price_moa += total_price_per_model[model]
        
    row1 = pd.Series(total_input_token_len, name='input_token').to_frame().T
    row2 = pd.Series(total_output_token_len, name='output_token').to_frame().T
    row3 = pd.Series(total_price_per_model, name='total_price_per_model').to_frame().T
    price = pd.concat([price, row1, row2, row3])
    price.loc['total_price_moa'] = [total_price_moa] + [np.nan] * (price.shape[1] - 1)

    price.to_csv(args.output_file)
    
