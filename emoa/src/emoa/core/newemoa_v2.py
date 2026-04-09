import json
import os
import time
import numpy as np
import asyncio
from fire import Fire
from textwrap import dedent
from pydantic import BaseModel
from multiprocessing import Pool
from typing import List, Dict, Optional
from loguru import logger

import re
from textwrap import dedent
from typing import Dict, List, Tuple
from emoa.utils import (
    get_eval_set
)
from emoa.utils.multiprocess import starstarmap
from emoa.core import EmoaPipeline
from emoa.serve.models import openai_api_generate, ModelAPIProvider
import torch
from torch import nn
import re, json, ast
from typing import Tuple, List

from transformers import AutoTokenizer
import pandas as pd

DEBUG = int(os.environ.get("DEBUG", "0"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MISSING_PEER_SENTINEL = -1.0
STOP_THRESHOLD = 0.805
K_MODEL_SELECT_THRESHOLD = 3
ROUTER_WEIGHT, SELF_WEIGHT, CROSS_WEIGHT = 2, 1, 1

class ChatCompletionMessage:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, message):
        self.message = message

class ChatCompletionResponse:
    def __init__(self, answer):
        self.choices = [Choice(message=ChatCompletionMessage(content=answer))]

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    extra_body: Optional[Dict] = {}



FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.I)   # ```json … ```
BRACE = re.compile(r"\{[\s\S]*?}", re.M)                    # Any {...}



def safe_json_extract(raw: str) -> Tuple[str, float, List[float]]:
    """
    Try to extract (answer, self_score, peer_scores) from raw.
    Supports single/double quotes, trailing commas, missing peer_scores,
    and even unquoted answer fields.
    If parsing fails, return (raw.strip(), 0.0, []).
    """
    # Internal parse attempts
    def try_parse(txt: str):
        txt = txt.strip().replace("{{", "{").replace("}}", "}")
        txt = re.sub(r",\s*([}\]])", r"\1", txt)          # Remove trailing commas
        txt = txt.replace("'", '"')                       # Single -> double quotes
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(txt)
                if isinstance(obj, dict):
                    ans  = obj.get("answer", "").strip()
                    self = float(obj.get("self_score", 0.0))
                    peer = [float(x) for x in obj.get("peer_scores", [])]
                    return raw.strip(), self, peer
            except Exception:
                continue
        return None

    
    for m in FENCE.finditer(raw):
        res = try_parse(m.group(1))
        if res: return res

    
    for m in BRACE.finditer(raw):
        res = try_parse(m.group(0))
        if res: return res

    
    ans_m  = re.search(r'["\']?answer["\']?\s*:\s*("?)([^,"\n}]+)\1', raw, re.I)
    self_m = re.search(r'["\']?self_score["\']?\s*:\s*([0-9]*\.?[0-9]+)', raw, re.I)
    if ans_m or self_m:
        ans  = (ans_m.group(2).strip() if ans_m else "").strip('"').strip("'")
        self = float(self_m.group(1)) if self_m else 0.0
        return raw.strip(), self, []


    return raw.strip(), 0.0, []


def remove_mcq_prefix(text: str) -> str:
    
    PREFIX = (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering."
    )
    
    if text.startswith(PREFIX):
        # After removing the prefix, also strip extra leading whitespace/newlines
        return text[len(PREFIX):].lstrip()
    else:
        # If the prefix is not matched, return the original text
        return text


def build_prompt_round_one() -> str:
    return dedent("""
        You are participating in a multi-agent reasoning task.

        **Your objectives**
        1. Produce the best possible answer to the user's query.
        2. Critically evaluate your own answer and give it a quality score **between 0 and 1**  
           (0 = completely wrong, 1 = perfect).

        **Output format** - return **ONLY** a valid JSON object:
        ```json
        {
          "answer": "<your answer>",
          "self_score": <float between 0 and 1>
        }
        ```
        Do **not** add any keys, comments or extra text.
    """)



def build_prompt_subsequent(previous_answers: List[str]) -> str:
    """
    Build the system prompt for round 2.
    `previous_answers` are enumerated as ANSWER_0, ANSWER_1, ... in the prompt.
    """
    answer_block = "\n\n".join(
        f"ANSWER_{i}: {ans}" for i, ans in enumerate(previous_answers)
    )
    return dedent(f"""
        You are participating in a multi-agent reasoning task.Here are several answers from other LLMs:

        {answer_block}

        **Your objectives**
        1. Taking every answer in the previous round into account and produce an improved answer to the user's query.
        2. Critically evaluate your own answer and give it a quality score **between 0 and 1**  
            (0 = completely wrong, 1 = perfect).
        3. Critically evaluate **each** ANSWER_i above with a value in [0, 1] representing its quality.
            (0 = completely wrong, 1 = perfect)

        **Output format** - return **ONLY** a valid JSON object:
        ```json
        {{
          "answer": "<your improved answer>",
          "self_score": <float>,
          "peer_scores": [<float_score_for_ANSWER_0>, <float_score_for_ANSWER_1>, ...]
        }}
        ```
        Do not include any other text.


    """)


class EmoaApp:

    def __init__(
            self,
            model: str,
            messages: List[ChatMessage],
            output_path: Optional[str] = None,
            candidate_models: Optional[List[str]] = None,
            temperature: Optional[float] = 0.7,
            max_tokens: Optional[int] = 2048,
            rounds: Optional[int] = 1,
            stream: bool = False,
            mode: str = "together",
            router: nn.Module = None,
            router_tokenizer: AutoTokenizer = None,
            api_info: str = None,
            agg_model: str = None,
            router_type: str = None,
    ) -> None:

        self.model = model
        self.output_path = output_path
        self.messages = messages

        self.candidate_models = candidate_models or []
        self.agg_model = agg_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rounds = rounds

        self.stream = stream
        self.mode = mode or "together"

        self.router = router
        self.router_tokenizer = router_tokenizer
        df = pd.read_csv(api_info)
        self.api_info = df.set_index('model').to_dict(orient='index')
        for key, value in self.api_info.items():
            for column, column_value in value.items():
                if isinstance(column_value, float) and np.isnan(column_value):
                    value[column] = None
        self.router_type = router_type

        self.emoa_pipeline = EmoaPipeline()

    def map_scores(self, scores, a=0.8, b=1.5, c=0.8, d=1.5):
        """
        Apply a power-function mapping to raw performance scores:
        For x < 0.5, use a * x^b
        For x >= 0.5, use 1 - c * (1 - x)^d
        """
        mapped = {}
        for model, x in scores.items():
            if 0 <= x < 0.5:
                mapped[model] = a * (x ** b)
            elif 0.5 <= x <= 1:
                mapped[model] = 1 - c * ((1 - x) ** d)
            else:
                raise ValueError(f"Score {x} out of range [0,1]")
        return mapped

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-max normalise a dict of {model: score} to the range [0, 1].

        If all scores are identical or the dict is empty, every value is set to 0.5,
        so that downstream weighting keeps them neutral but non-zero.
        """
        if not scores:
            return {}
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 0.5 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    async def run(
            self,
    ):
        logger.info(f"Start.")
        process_func = self._get_process_function(mode=self.mode)
        ret_list = await process_func({"messages": self.messages})

        return ret_list

    async def process_func_emoa(
            self,
            item,
    ):

        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0] * len(self.candidate_models))
            ),
            completion_tokens=dict(
                zip(self.candidate_models, [0] * len(self.candidate_models))
            ),
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

       #----------------------------------------
        if self.router_type == "routerdc":
            # 1. Compute raw model scores


            tokens = self.router_tokenizer(
                remove_mcq_prefix(user_message),
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            with torch.no_grad():
                logits, _ = self.router(t=1, input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
            
            print(probs)
    
            probs_0 = probs[::2]
            #performances_0 = {'ContactDoctor/Bio-Medical-Llama-3-8B': 0, 'Qwen/Qwen2.5-Coder-7B-Instruct': 1, 'Qwen/Qwen2.5-Math-7B-Instruct': 1, 'google/gemma-2-9b-it': 0, 'mistralai/Ministral-8B-Instruct-2410': 1}
            performances = {m: p for m, p in zip(self.candidate_models, probs_0)}    
            print(performances)
            # breakpoint()       
            
            sorted_models = sorted(
                self.candidate_models,
                key=lambda m: (
                    -performances[m],  # Higher raw score first
                    self.api_info[m]['output_price'],  # Lower output price first
                    self.api_info[m]['input_price'],  # Lower input price first
                    # self.api_info[m]['latency']             # Lower latency first
                )
            )


            print("————————————————————————————————————————")
            print("init performance score :", performances)


            # 3. Use the top-ranked model as the summarizer (agg_model)
            best_anwer_model = sorted_models[0]
            

            # 4. Apply power-function mapping to all model scores
            map_pref_0 = self.map_scores(
                performances, a=0.8, b=1.5, c=0.8, d=1.5
            )
            

            # Maintain an overall aggregated‑score dictionary across all models
            # aggregated_scores = {m: map_pref_0[m] for m in self.candidate_models}
            aggregated_scores = {m: map_pref_0[m] for m in self.candidate_models}

            
            print("mapped performance score 0:", map_pref_0)
        
            # print("mapped performance score sum:", aggregated_scores)

            # 5. Select models whose mapped-score gap to agg_model is <= 0.2
            threshold = 0.2
            selected_models = [
                m for m in sorted_models
                if (map_pref_0[best_anwer_model] - map_pref_0[m]) <= threshold
            ]

            # 6. If selected models > 3, keep only the top 3 by ranking; otherwise keep all
            #reference_models = selected_models[:3] if len(selected_models) > 3 else selected_models
            reference_models = selected_models[:K_MODEL_SELECT_THRESHOLD] if len(selected_models) > K_MODEL_SELECT_THRESHOLD else selected_models

        # -------------------------Model collaboration stage--------------------

        orig_system_prompt = None
        if messages[0]["role"] == "system":
            orig_system_prompt = messages[0]["content"]

        references = item.get("references", [])

        emoa_pipeline = self.emoa_pipeline

        round_answers: List[Dict[str, str]] = []
        round_self_scores: List[Dict[str, float]] = []
        round_peer_scores: List[Dict[str, List[float]]] = []

        current_models = reference_models.copy()
        current_models_prev: List[str] = []
        

        # ---------------------Start iterative collaboration---------------
        for i_round in range(3):

            if i_round == 0:
                # round 1

                
                print("Round 1")

                prev_refs = None
                system_prompt = build_prompt_round_one()
                
            elif i_round == 1:
                # round 2
                
                print("Round 2")

                prev_refs = [round_answers[0][m] for m in current_models_prev]
                system_prompt = build_prompt_subsequent(prev_refs)

                # print("----------------------")
                # print(prev_refs)
                # print("----------------------")

            else:
                # round 3
                
                print("Round 3")

                answer_block = "\n\n".join(
                    f"ANSWER_{i}: {round_answers[1][m]}"
                    for i, m in enumerate(current_models_prev)
                )
                system_prompt = dedent(f"""
                    You are participating in a multi-agent reasoning task.
                    Here are some answers generated by LLMs in the last round:
                    {answer_block}

                    **Your objective**
                    Produce the best possible answer to the user’s query, taking into
                    account every ANSWER_i above.  Return only your answer text.

                    **Output format - return ONLY the raw answer (no JSON or any other file).**

                """)
                prev_refs = [round_answers[1][m] for m in current_models_prev]

                

            # ----------------answer generate for the round ------------
            async_calls = [
                emoa_pipeline.generate_with_references(
                    model=m,
                    messages=messages,
                    references=None,
                    # references=prev_refs,
                    orig_system_prompt=orig_system_prompt,
                    system_prompt_to_inject=system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_info=self.api_info,
                )
                for m in current_models
            ]
            responses = await asyncio.gather(*async_calls)

            answers, self_scores, peer_scores = {}, {}, {}
            for m, resp in zip(current_models, responses):
                raw = resp.choices[0].message.content.strip()

                # Round‑3 returns plain text; earlier rounds return JSON
                if i_round < 2:
                    ans, ss, ps = safe_json_extract(raw)
                    self_scores[m] = ss
                    peer_scores[m] = ps
                else:
                    ans, ss, ps = raw, 0.0, []

                answers[m] = ans

                bill["prompt_tokens"][m] += resp.usage.prompt_tokens
                bill["completion_tokens"][m] += resp.usage.completion_tokens

            round_answers.append(answers)
            round_self_scores.append(self_scores)
            round_peer_scores.append(peer_scores)

            if i_round == 2:
                break

           
            # print("Round Self Scores", round_self_scores)
            # print("Round Peer Scores", round_peer_scores)

            
            # print("Current_models:", current_models)

            current_models_prev = current_models.copy()

            print(f"This is round {i_round + 1}")
            print(round_answers[i_round])

            cross_scores: Dict[str, float] = {}

            if i_round == 1:
                try:

                    has_valid_peer = (
                            len(round_peer_scores) > 1 and
                            any(round_peer_scores[1].get(m) for m in round_peer_scores[1])
                    )

                    if has_valid_peer:
                        best_model_r2 = max(self_scores, key=self_scores.get)
                        cross_scores = {
                            model_name: (
                                round_peer_scores[1][best_model_r2][idx]
                                if idx < len(round_peer_scores[1][best_model_r2])
                                else MISSING_PEER_SENTINEL
                            )
                            for idx, model_name in enumerate(reference_models)
                        }

                        print("valid peer scores")
                    else:

                        cross_scores = {m: MISSING_PEER_SENTINEL for m in reference_models}
                except Exception:

                    cross_scores = {m: MISSING_PEER_SENTINEL for m in reference_models}

            # print("!!!!!!!!!!!")
            # print("cross_scores:", cross_scores)

            # aggregated score for selection
            agg_score = {}

            for m in current_models_prev:

                # round 1 socre
                if i_round == 0:
                    W_ROUTER = ROUTER_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT)
                    W_SELF = SELF_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT)
                    router_part = map_pref_0[m]
                    self_part = self_scores.get(m, 0.0)

                    if self_part == 0:
                        self_part = 0.8
                    
                    agg_score[m] = (
                                router_part * W_ROUTER
                                + self_part * W_SELF
                        )


                # round 2 score
            if i_round == 1:
                for m in self.candidate_models:
                    if m in current_models_prev and m in reference_models:

                        W_ROUTER = ROUTER_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        W_SELF0 = (SELF_WEIGHT/2)/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        W_SELF1 = (SELF_WEIGHT/2)/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        W_CROSS = CROSS_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        router_part = map_pref_0[m]
                        self_part0 = 0.5 if self_scores.get(m, 0.0) == 0 else self_scores.get(m, 0.0)
                        self_part1 = 0.5 if self_scores.get(m, 1.0) == 0 else self_scores.get(m, 1.0)
                        cross_part = cross_scores.get(m, 1.0)

                        
                        if cross_part == MISSING_PEER_SENTINEL:
                            agg_score[m] = aggregated_scores[m]

                        else:
                            agg_score[m] = (
                                    router_part * W_ROUTER
                                    + self_part0 * W_SELF0
                                    + self_part1 * W_SELF1
                                    + cross_part * W_CROSS
                            )
                    elif m in current_models_prev and m not in reference_models:
                        W_ROUTER = ROUTER_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT)
                        W_SELF1 = SELF_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT)
                        router_part = map_pref_0[m]
                        
                        self_part1 = 0.5 if self_scores.get(m, 1.0) == 0 else self_scores.get(m, 1.0)
                    
                        agg_score[m] = (
                                router_part * W_ROUTER
                                + self_part1 * W_SELF1
                        )
                    
                    else:
                        W_ROUTER = ROUTER_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        W_SELF0 = SELF_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        W_CROSS = CROSS_WEIGHT/(ROUTER_WEIGHT+SELF_WEIGHT+CROSS_WEIGHT)
                        router_part = map_pref_0[m]
                        self_part0 = 0.3 if self_scores.get(m, 0.0) == 0 else self_scores.get(m, 0.0)
                        cross_part = cross_scores.get(m, 1.0)

                        
                        if cross_part == MISSING_PEER_SENTINEL:
                            agg_score[m] = aggregated_scores[m]

                        else:
                            agg_score[m] = (
                                    router_part * W_ROUTER
                                    + self_part0 * W_SELF0
                                    + cross_part * W_CROSS
                            )

            # ---- update the global aggregated_scores only for models that ran this round ----
            for m, sc in agg_score.items():
                aggregated_scores[m] = sc

            # print("!!!!!!!!!!!!!")
            # print("aggscore:", aggregated_scores)

            K = min(K_MODEL_SELECT_THRESHOLD, len(selected_models))  
            current_models = sorted(
                selected_models,
                key=lambda m: -aggregated_scores.get(m, 0.0)
            )[:K]
            print("current_models:", current_models)
            
            #agg_model = agg_model = max(aggregated_scores, key=aggregated_scores.get)
            #agg_model = sorted_models[0]
            #agg_model = agg_sorted_models[0]
            agg_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
            print(agg_model)


            print("current_models:", current_models)

            ##---------------- Early‑stop ----------------
            best_model, best_score = max(agg_score.items(), key=lambda kv: kv[1])
            
            if best_score > STOP_THRESHOLD : 
                
                break
            
        last_round_models = list(round_answers[-1].keys())
        final_references   = [round_answers[-1][m] for m in last_round_models]

        print(f"final_references: {final_references}")
        # breakpoint()
        
        
        # ========= Skip summarisation if only one model remains =========
        # if len(current_models) == 1:
        #     best_model = current_models[0]
        #     agg_model = best_model
        #     chosen_answer = final_references[0]

        #     resp_obj = ChatCompletionResponse(chosen_answer)

        #     end_time = time.time()

        #     # calculate cost for all models with the tokens already accrued
        #     for model in self.candidate_models:
        #         bill['cost'] += (
        #                 self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1_000_000
        #                 + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1_000_000
        #         )

        #     return {
        #         "id": None,
        #         "object": "chat.completion",
        #         "created": time.time(),
        #         "model": best_model,
        #         "agg_model": agg_model,
        #         "choices": resp_obj.choices,                      # Keep response structure consistent
        #         "final_answer": resp_obj.choices[0].message.content,
        #         "bill": bill,
        #         "latency": end_time - start_time,
        #     }

        # --------------------------Summarization stage----------------------
        agg_prompt = dedent("""
        You have been provided with a set of responses from various open-source
        models to the latest user query. Your task is to synthesise these
        responses into a single, high-quality answer. Critically evaluate the
        information given, correct any mistakes, and produce a coherent,
        well-structured response that meets the highest standards of accuracy.

        Responses from models:
    """)
        
        
        print("!!!!!!!!!!!!!")
        print("agg model:", agg_model)
        print("round num:", i_round)
        summary_resp = await emoa_pipeline.generate_with_references(
            model=agg_model,
            messages=messages,
            references=final_references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=agg_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
        )

        end_time = time.time()

        bill['prompt_tokens'][agg_model] += summary_resp.usage.prompt_tokens
        bill['completion_tokens'][agg_model] += summary_resp.usage.completion_tokens

        for model in self.candidate_models:
            bill['cost'] += (
                    self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1_000_000
                    + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1_000_000
            )

        return {
            "id": None,
            "object": "chat.completion",
            "created": time.time(),
            "model": self.model,
            "agg_model": agg_model,
            "choices": summary_resp.choices,
            "final_answer": summary_resp.choices[0].message.content,
            "bill": bill,
            "latency": end_time - start_time,
        }

    def _get_process_function(self, mode: str = "naive"):
        return self.process_functions[mode]

    @property
    def process_functions(self) -> Dict:
        return {
            "emoa": self.process_func_emoa
        }


if __name__ == "__main__":
    Fire(EmoaApp)
