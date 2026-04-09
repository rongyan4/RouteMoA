"""
SMoA: Sparse-Mixture-of-Agents implementation
══════════════════════════════════════════════
• Role-Playing
• Response Selection (Judge)
• Early Stopping (Moderator)
"""

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

from emoa.utils import (
    get_eval_set
)

from emoa.utils.multiprocess import starstarmap
from emoa.core import EmoaPipeline
from emoa.serve.models import openai_api_generate, ModelAPIProvider
import torch
from torch import nn
from transformers import AutoTokenizer
import pandas as pd

import random
import re
from typing import Union


def read_file(path: Optional[str], default_content: Union[str, dict]) -> Union[str, dict]:
    if path and os.path.isfile(path):
        _, ext = os.path.splitext(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                if ext.lower() == ".json":
                    return json.load(f)
                elif ext.lower() == ".txt":
                    return f.read()
                else:
                    return default_content
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return default_content
    return default_content


def parse_judge_output(text: str):
    """Return (indexes, stop_flag) extracted from a judge‑style JSON answer."""
    idx_match  = re.search(r'"chosen responses"\s*:\s*(\[[^\]]+\])', text)
    stop_match = re.search(r'"end debate"\s*:\s*(true|false)', text, re.I)

    # Try to load the index list if it exists and is valid JSON
    try:
        idx = json.loads(idx_match.group(1)) if idx_match else None
    except Exception:
        idx = [0]

    # Fallback: if nothing (or an empty list) was parsed, default to [0]
    if not idx:
        idx = [0]

    # Stop flag: defaults to False unless we explicitly see `"end debate": true`
    stop = bool(stop_match and stop_match.group(1).lower() == "true")
    return idx, stop


def extract_role_descriptions(text: str) -> List[str]:
    """
    Split the raw role generation output into individual role description blocks.

    The generator marks each role with “[Generated Role Description]”.  We split
    on this marker, strip whitespace, and discard empty fragments.
    """
    parts = re.split(r'\[Generated Role Description\]', text)
    return [p.strip() for p in parts if p.strip()]


DEBUG = int(os.environ.get("DEBUG", "0"))


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


class SmoaApp:

    def __init__(
            self,
            model: str,
            messages: List[ChatMessage],
            output_path: Optional[str] = None,
            candidate_models: Optional[List[str]] = None,
            temperature: Optional[float] = 0.7,
            max_tokens: Optional[int] = 2048,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            rounds: Optional[int] = 1,
            stream: bool = False,
            router: nn.Module = None,
            router_tokenizer: AutoTokenizer = None,
            api_info: str = None,
            agg_model: str = None,
            router_type: str = None,
            add_role: bool = False,
            moderate_select: bool = False,
            moderate_end: bool = False,
            num_select_response: int = 3,
            role_prompt_file: str = None,
            judge_prompt_file: str = None,
            aggregation_prompt_file: str = None,
            extra_headers: Dict = None,
            **kwargs
    ) -> None:

        self.model = model
        self.output_path = output_path
        self.messages = messages

        self.candidate_models = candidate_models or []
        self.agg_model = agg_model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rounds = rounds

        self.stream = stream

        self.router = router
        self.router_tokenizer = router_tokenizer

        # --- SMoA flags ---
        self.add_role = add_role
        self.moderate_select = moderate_select
        self.moderate_end = moderate_end
        self.k = max(1, num_select_response)
        self.extra_headers = extra_headers or {}
        # --- prompt templates ---
        self.role_template = read_file(
            role_prompt_file,
            """Your task is to generate {n} role descriptions for the following task:
        {task}, Each role description should include occupation, personality, and social group. 
        Separate each role description with [Generated Role Description]"""
        )

        self.judge_template = read_file(
            judge_prompt_file,
            """You will be given {n} candidate responses to the user's question.

        Your answer should be like this JSON FILE STYLE:
        {{
        "chosen responses": [indexes of best {k} responses],
        "end debate": true/false
        }}"""
        )

        self.aggregation_prompt = read_file(
            aggregation_prompt_file,
            dedent("""\
            You are given several candidate answers from different open‑source models.
            Produce a single, high‑quality answer. Do not copy verbatim; analyse,
            compare and write a concise, accurate response.
            Responses:
            """).strip(),
        )
        self.role_descriptions: List[str] = []

        df = pd.read_csv(api_info)
        self.api_info = df.set_index('model').to_dict(orient='index')
        for key, value in self.api_info.items():
            for column, column_value in value.items():
                if isinstance(column_value, float) and np.isnan(column_value):
                    value[column] = None
        if not self.candidate_models:
            self.candidate_models = list(self.api_info.keys())

        self.smoa_pipeline = EmoaPipeline()

    async def run(
            self,
    ):
        logger.info(f"Start.")
        process_func = self._get_process_function(model=self.model)
        ret_list = await process_func({"messages": self.messages})

        return ret_list

    async def process_func_smoa(self, item):
        bill = self.init_bill()
        start_time = time.time()
        messages = item["messages"]
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        print("user_message:", user_message)

        # --- choose reference & aggregator models (placeholder – keep hard‑coded) ---
        reference_models = self.candidate_models   #random.sample(self.candidate_models, 5)
        agg_model = self.agg_model

        # --- role generation ---
        await self.generate_roles(user_message)

        references: List[str] = []
        prev_references: List[str] = []

        for i_round in range(self.rounds):
            # -- collect candidate answers --
            tasks = []

            if i_round == 0:
                for m in reference_models:
                    role_prompt = ""
                    if self.add_role and self.role_descriptions:
                        role_prompt = self.role_descriptions[reference_models.index(m)]
                        print("role_prompt:", role_prompt)
                    tasks.append(
                        self.smoa_pipeline.generate_with_references(
                            model=m,
                            messages=messages,
                            references=None,
                            system_prompt_to_inject=role_prompt,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            timeout=self.timeout,
                            max_tokens=self.max_tokens,
                            api_info=self.api_info,
                            extra_headers=self.extra_headers,
                        )
                    )

                    print("messages:", messages)
                    print("role_prompt:", role_prompt)
            
            else:
                for m in reference_models:
                    role_prompt = ""
                    if self.add_role and self.role_descriptions:
                        role_prompt = self.role_descriptions[reference_models.index(m)]
                        print("role_prompt:", role_prompt)
                    tasks.append(
                        self.smoa_pipeline.generate_with_references(
                            model=m,
                            messages=messages,
                            references=prev_references,
                            system_prompt_to_inject=role_prompt,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            timeout=self.timeout,
                            max_tokens=self.max_tokens,
                            api_info=self.api_info,
                            extra_headers=self.extra_headers,
                        )
                    )

                    print("messages:", messages)
                    print("role_prompt:", role_prompt)

            resps = await asyncio.gather(*tasks)

            references = []
            for m, resp in zip(reference_models, resps):
                try:
                    txt = resp.choices[0].message.content.strip()
                except:
                    txt = ""
                    logger.warning(f"Failed to get response from model {m}")
                if txt:
                    references.append(txt)
                    self.add_usage(bill, m, resp)

            # -- moderation / early‑stop --
            references, stop_flag = await self.judge_and_moderate(references, user_message)
            if stop_flag:
                break
            prev_references = references

        # --- aggregation step ---
        agg_resp = await self.smoa_pipeline.generate_with_references(
            model=agg_model,
            messages=messages,
            references=references,
            system_prompt_to_inject=self.aggregation_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
            extra_headers=self.extra_headers,
        )
        self.add_usage(bill, agg_model, agg_resp)

        return {
            "id": None,
            "object": "chat.completion",
            "created": time.time(),
            "model": self.model,
            "agg_model": agg_model,
            "choices": agg_resp.choices,
            "bill": bill,
            "latency": time.time() - start_time,
        }

    def _get_process_function(self, model: str = "smoa"):
        return self.process_functions[model]

    @property
    def process_functions(self) -> Dict:
        return {
            "smoa": self.process_func_smoa
        }

    # =================== billing helpers ===================
    def init_bill(self):
        return dict(
            prompt_tokens=dict.fromkeys(self.candidate_models, 0),
            completion_tokens=dict.fromkeys(self.candidate_models, 0),
            cost=0.0,
        )

    def add_usage(self, bill, model, resp):
        # make sure the model key exists
        if model not in bill["prompt_tokens"]:
            bill["prompt_tokens"][model] = 0
            bill["completion_tokens"][model] = 0
        # accumulate token counts
        bill["prompt_tokens"][model] += resp.usage.prompt_tokens
        bill["completion_tokens"][model] += resp.usage.completion_tokens
        # cost = price * *incremental* tokens (reference: SMoA implementation)
        bill["cost"] += (
                self.api_info[model]["input_price"] * resp.usage.prompt_tokens / 1e6
                + self.api_info[model]["output_price"] * resp.usage.completion_tokens / 1e6
        )

    # =================== role generation ===================
    async def generate_roles(self, task_desc: str):
        if self.role_descriptions or not self.add_role:
            return
        prompt = self.role_template.format(n=len(self.candidate_models), task=task_desc)

        print("generate_role_prompt:", prompt)
        model = self.agg_model or self.candidate_models[0]
        print("role_generation_model:", model)
        resp = await self.smoa_pipeline.generate_with_references(
            model=self.agg_model or self.candidate_models[0],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            timeout=self.timeout,
            api_info=self.api_info,
            extra_headers=self.extra_headers,
        )
        self.role_descriptions = extract_role_descriptions(resp.choices[0].message.content)
        # --- length‑sanity: keep the list aligned with candidate models ---
        n_models = len(self.candidate_models)
        if n_models:
            if len(self.role_descriptions) > n_models:
                # truncate extra roles
                self.role_descriptions = self.role_descriptions[:n_models]
            elif len(self.role_descriptions) < n_models:
                # pad missing roles with a generic helper prompt
                self.role_descriptions.extend(
                    ["You are a helpful assistant."] * (n_models - len(self.role_descriptions))
                )
        logger.info(f"Role descriptions: {self.role_descriptions}")

    # =================== judge / moderation ===================
    async def judge_and_moderate(self, refs: List[str], question: str):
        agg_model = self.agg_model 

        if not (self.moderate_select or self.moderate_end):
            return refs, False
        prompt = self.judge_template.format(n=len(refs), k=self.k, question=question)
        for i, r in enumerate(refs):
            prompt += f"\nResponse {i}:\n{r}\n"

        print("judge_and_moderate_prompt:", prompt)

        resp = await self.smoa_pipeline.generate_with_references(
            model=agg_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            top_p=0.9,
            timeout=self.timeout,
            api_info=self.api_info,
            extra_headers=self.extra_headers,
        )

        print("=== raw judge response ===")
        print(resp.choices[0].message.content)

        chosen, stop = parse_judge_output(resp.choices[0].message.content)

        print("=== parsed judge response ===")
        print("chosen:", chosen, "stop:", stop)

        if not chosen or any(i >= len(refs) for i in chosen):
            chosen = list(range(min(self.k, len(refs))))
        refs = [refs[i] for i in chosen]
        return refs, (self.moderate_end and stop)


if __name__ == "__main__":
    Fire(SmoaApp)
