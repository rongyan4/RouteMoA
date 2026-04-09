import json
import os
import time
import numpy as np
import asyncio
from fire import Fire
from textwrap import dedent
from pydantic import BaseModel
from multiprocessing import Pool
from typing import List, Dict, Optional, AsyncGenerator
from loguru import logger
import random
from fastapi.responses import StreamingResponse
import tiktoken

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

def chunk_to_dict(chunk, model_override=None):
    return {
        "id": chunk.id,
        "object": "chat.completion.chunk",
        "created": chunk.created,
        "model": model_override or chunk.model,
        "choices": [
            {
                "index": c.index,
                "delta": c.delta.model_dump(exclude_unset=True),
                "logprobs": getattr(c, "logprobs", None),
                "finish_reason": c.finish_reason,
            }
            for c in chunk.choices
        ],
        "usage": chunk.usage.model_dump() if getattr(chunk, "usage", None) else None,
    }

DEBUG = int(os.environ.get("DEBUG", "0"))
class ChatMessage(BaseModel):
    role: str
    content: str

class EmoaApp:

    def __init__(
            self,
            model: str,
            messages: List[ChatMessage],
            output_path: Optional[str] = None,
            candidate_models: Optional[List[str]] = None,
            temperature: Optional[float] = 0.7,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            max_tokens: Optional[int] = 2048,
            rounds: Optional[int] = 1,
            router: nn.Module = None,
            router_tokenizer: AutoTokenizer = None,
            api_info: str = None,
            extra_headers: Dict = None,
            agg_model: str = None,
            router_threshold: float = 0.0625,
            router_top_k: int = 3,
            stream: Optional[bool] = False,
            **kwargs
        ) -> None:
        
        # 处理可能重复的参数
        if 'stream' in kwargs:
            # 忽略 kwargs 中的 stream，使用参数中的值
            del kwargs['stream']
        self.model = model
        self.output_path = output_path
        self.messages = messages
        self.stream = stream

        self.candidate_models = candidate_models or []
        self.agg_model = agg_model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rounds = rounds
        self.router = router
        self.router_tokenizer = router_tokenizer
        df = pd.read_csv(api_info)
        self.api_info = df.set_index('model').to_dict(orient='index')
        for key, value in self.api_info.items():
            for column, column_value in value.items():
                if isinstance(column_value, float) and np.isnan(column_value):
                    value[column] = None
        if not self.candidate_models:  # 没指定candidate_models则选择全部模型
            self.candidate_models = list(self.api_info.keys())
        
        self.extra_headers = extra_headers or {}
        self.emoa_pipeline = EmoaPipeline()
        self.router_threshold = router_threshold
        self.router_top_k = router_top_k


    async def run(self):
        process_func = self._get_process_function(model=self.model)
        logger.info(f"Start non-streaming processing.")
        result = await process_func({"messages": self.messages})
        return result
    
    async def run_stream(self):
        process_func = self._get_process_function_stream(model=self.model)
        logger.info(f"Start streaming processing.")
        async for chunk in process_func({"messages": self.messages}):
            yield chunk

    async def process_func_router(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_message = msg["content"]
                else:   # 用于处理这种输入：'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '请将“Beauty is in the eye of the beholder”翻译为中文'}]}]
                    user_message = msg["content"][0]["text"]
                break
        print("**************** Routing Start ****************")
        tokens = self.router_tokenizer(
            user_message,
            max_length=512,
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to("cuda")
        attention_mask = tokens["attention_mask"].to("cuda")

        with torch.no_grad():
            logits, _ = self.router(t=1, input_ids=input_ids, attention_mask=attention_mask)
        # probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        # 去掉第四个prob (Qwen2.5-14B-Instruct没有部署)
        probs.pop(3)
        performances = dict(zip(self.candidate_models, probs))
        
        print("init performance score 0:", performances)
        reference_models = sorted(
                self.candidate_models,
                key=lambda model: (
                    -performances[model],  # 按照得分从高到低排序（负号表示降序）
                )
            )
        reference_models = reference_models[:self.router_top_k]
        reference_models = sorted(
            reference_models,
            key=lambda model: (
                self.api_info[model]['output_price'],  # 按照 output_price 从低到高排序
                self.api_info[model]['input_price']  # 按照 input_price 从低到高排序
            )
        )
        model = reference_models[0]
        
        print(
            f"""
            target model: {model}
            prob: {performances[model]}
            """
        )
        print("**************** Generation Start ****************")
        model_api_provider = ModelAPIProvider(api_info=self.api_info)
        
        response = await model_api_provider.generate(
            model=model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            extra_headers=self.extra_headers,
        )

        end_time = time.time()
        bill['prompt_tokens'][model] += response.usage.prompt_tokens
        bill['completion_tokens'][model] += response.usage.completion_tokens

        for model in self.candidate_models:
            bill['cost'] += self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1000000 \
            + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1000000

        return {
            "id": None,
            "object": "chat.completion",
            "created": time.time(),
            "model": self.model,
            "choices": response.choices,
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": [model], "aggregator": None}
        }
        

    async def process_func_router_stream(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_message = msg["content"]
                else:   # 用于处理这种输入：'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '请将“Beauty is in the eye of the beholder”翻译为中文'}]}]
                    user_message = msg["content"][0]["text"]
                break
        print("**************** Routing Start ****************")
        tokens = self.router_tokenizer(
            user_message,
            max_length=512,
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to("cuda")
        attention_mask = tokens["attention_mask"].to("cuda")

        with torch.no_grad():
            logits, _ = self.router(t=1, input_ids=input_ids, attention_mask=attention_mask)
        # probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        # 去掉第四个prob (Qwen2.5-14B-Instruct没有部署)
        probs.pop(3)
        performances = dict(zip(self.candidate_models, probs))
        
        print("init performance score 0:", performances)
        reference_models = sorted(
                self.candidate_models,
                key=lambda model: (
                    -performances[model],  # 按照得分从高到低排序（负号表示降序）
                )
            )
        reference_models = reference_models[:self.router_top_k]
        reference_models = sorted(
            reference_models,
            key=lambda model: (
                self.api_info[model]['output_price'],  # 按照 output_price 从低到高排序
                self.api_info[model]['input_price']  # 按照 input_price 从低到高排序
            )
        )
        model = reference_models[0]
        
        print(
            f"""
            target model: {model}
            prob: {performances[model]}
            """
        )
        print("**************** Generation Start ****************")
        model_api_provider = ModelAPIProvider(api_info=self.api_info)
        final_usage = None

        async for chunk in model_api_provider.generate_stream(
            model=model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            extra_headers=self.extra_headers,
            stream=True
        ):
            # 如果 OpenAI 返回了 finish_reason，就记录 usage
            if chunk.choices[0].finish_reason is not None:
                # breakpoint()
                final_usage = chunk.usage
                break
            # 把 chunk 转成普通 dict
            chunk_dict = chunk_to_dict(chunk, self.model)
            chunk_dict["bill"] = None
            chunk_dict["latency"] = None
            chunk_dict["routed_model"] = {"proposer": [model], "aggregator": None}
            # yield JSON 字符串，SSE 格式
            yield f"data: {json.dumps(chunk_dict)}\n\n"

        # 统计费用等
        end_time = time.time()
        if final_usage is not None:
            bill['prompt_tokens'][model] += final_usage.prompt_tokens
            bill['completion_tokens'][model] += final_usage.completion_tokens
        else:
            logger.warning("final_usage is None, skipping token billing for agg_model")
        bill['cost'] = sum(
            self.api_info[m]['input_price'] * bill['prompt_tokens'][m] / 1e6 +
            self.api_info[m]['output_price'] * bill['completion_tokens'][m] / 1e6
            for m in self.candidate_models
        )

        # 最后发送 summary chunk
        summary_chunk = {
            "id": None,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": self.model,
            "choices": [{
                "delta": {
                    "content": "",
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                    "refusal": None
                },
                "index": 0,
                "finish_reason": None,
                "logprobs": None
                }],
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": [model], "aggregator": None}
        }
        yield f"data: {json.dumps(summary_chunk)}\n\n"


    async def process_func_routemoa(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_message = msg["content"]
                else:
                    user_message = msg["content"][0]["text"]
                break

        tokens = self.router_tokenizer(
            user_message,
            max_length=512,
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to("cuda")
        attention_mask = tokens["attention_mask"].to("cuda")

        with torch.no_grad():
            logits, _ = self.router(t=1, input_ids=input_ids, attention_mask=attention_mask)
        # probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        # 去掉第四个prob (Qwen2.5-14B-Instruct没有部署)
        probs.pop(3)
        performances = dict(zip(self.candidate_models, probs))
        print("**************** Routing Start ****************")
        print("init performance score 0:", performances)
        reference_models = sorted(
                self.candidate_models,
                key=lambda model: (
                    -performances[model],  # 按照得分从高到低排序（负号表示降序）
                    self.api_info[model]['output_price'],  # 按照 output_price 从低到高排序
                    self.api_info[model]['input_price']   # 按照 input_price 从低到高排序
                )
            )
        reference_models = reference_models[:3]
        agg_model = max(performances, key=performances.get)
        print(
            f"""
            reference models: {reference_models}
            agg model: {agg_model}
            """
        )
        print("**************** Generation Start ****************")
        orig_system_prompt = None
        if messages[0]["role"] == "system":
            orig_system_prompt = messages[0]["content"]

        references = item.get("references", [])

        emoa_pipeline = self.emoa_pipeline

        if len(references) == 0 and len(reference_models) > 0:

            prev_references = []

            for i_round in range(self.rounds):
                if DEBUG:
                    logger.info(
                        f"Round {i_round+1}/{self.rounds} to collecting reference responses."
                    )

                async_responses = []
                for reference_model in reference_models:
                    async_response = emoa_pipeline.generate_with_references(
                        model=reference_model,
                        messages=messages,
                        references=prev_references,
                        orig_system_prompt=orig_system_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                        max_tokens=self.max_tokens,
                        api_info=self.api_info,
                        extra_headers=self.extra_headers
                    )
                    
                    async_responses.append(async_response)

                # Await all responses without blocking main process
                responses = await asyncio.gather(*async_responses)

                references = []
                for model, response in zip(reference_models, responses):
                    try:
                        ref = response.choices[0].message.content.strip()
                    except:
                        ref = ""
                        logger.warning(f"Failed to get response from model {model}")
                    if ref:
                        references.append(ref)
                        bill['prompt_tokens'][model] += response.usage.prompt_tokens
                        bill['completion_tokens'][model] += response.usage.completion_tokens

                if i_round < self.rounds - 1:
                    prev_references = references
                    references = []
                await asyncio.sleep(0.1)

        logger.info(f"{references}")

        agg_prompt = dedent("""
        You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

        Responses from models:
        """)
        response = await emoa_pipeline.generate_with_references(
            model=agg_model,
            messages=messages,
            references=references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=agg_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
            extra_headers=self.extra_headers
        )
        end_time = time.time()
        bill['prompt_tokens'][agg_model] += response.usage.prompt_tokens
        bill['completion_tokens'][agg_model] += response.usage.completion_tokens

        for model in self.candidate_models:
            bill['cost'] += self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1000000 \
            + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1000000

        return {
            "id": None,
            "object": "chat.completion",
            "created": time.time(),
            "model": self.model,
            "agg_model": agg_model,
            "choices": response.choices,
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": reference_models, "aggregator": agg_model}
        }


    async def process_func_routemoa_stream(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    user_message = msg["content"]
                else:
                    user_message = msg["content"][0]["text"]
                break

        tokens = self.router_tokenizer(
            user_message,
            max_length=512,
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to("cuda")
        attention_mask = tokens["attention_mask"].to("cuda")

        with torch.no_grad():
            logits, _ = self.router(t=1, input_ids=input_ids, attention_mask=attention_mask)
        # probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        # 去掉第四个prob (Qwen2.5-14B-Instruct没有部署)
        probs.pop(3)
        performances = dict(zip(self.candidate_models, probs))
        print("**************** Routing Start ****************")
        print("init performance score 0:", performances)
        reference_models = sorted(
                self.candidate_models,
                key=lambda model: (
                    -performances[model],  # 按照得分从高到低排序（负号表示降序）
                    self.api_info[model]['output_price'],  # 按照 output_price 从低到高排序
                    self.api_info[model]['input_price']   # 按照 input_price 从低到高排序
                )
            )
        reference_models = reference_models[:3]
        agg_model = max(performances, key=performances.get)
        print(
            f"""
            reference models: {reference_models}
            agg model: {agg_model}
            """
        )
        print("**************** Generation Start ****************")
        orig_system_prompt = None
        if messages[0]["role"] == "system":
            orig_system_prompt = messages[0]["content"]

        references = item.get("references", [])

        emoa_pipeline = self.emoa_pipeline

        if len(references) == 0 and len(reference_models) > 0:

            prev_references = []

            for i_round in range(self.rounds):
                if DEBUG:
                    logger.info(
                        f"Round {i_round+1}/{self.rounds} to collecting reference responses."
                    )

                async_responses = []
                for reference_model in reference_models:
                    async_response = emoa_pipeline.generate_with_references(
                        model=reference_model,
                        messages=messages,
                        references=prev_references,
                        orig_system_prompt=orig_system_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                        max_tokens=self.max_tokens,
                        api_info=self.api_info,
                        extra_headers=self.extra_headers
                    )
                    
                    async_responses.append(async_response)

                # Await all responses without blocking main process
                responses = await asyncio.gather(*async_responses)

                references = []
                for model, response in zip(reference_models, responses):
                    try:
                        ref = response.choices[0].message.content.strip()
                    except:
                        ref = ""
                        logger.warning(f"Failed to get response from model {model}")
                    if ref:
                        references.append(ref)
                        bill['prompt_tokens'][model] += response.usage.prompt_tokens
                        bill['completion_tokens'][model] += response.usage.completion_tokens

                if i_round < self.rounds - 1:
                    prev_references = references
                    references = []
                await asyncio.sleep(0.1)

        logger.info(f"{references}")

        agg_prompt = dedent("""
        You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

        Responses from models:
        """)
        final_usage = None

        async for chunk in emoa_pipeline.generate_with_references_stream(
            model=agg_model,
            messages=messages,
            references=references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=agg_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
            extra_headers=self.extra_headers
        ):
            # 如果 OpenAI 返回了 finish_reason，就记录 usage
            if chunk.choices[0].finish_reason is not None:
                final_usage = chunk.usage
                break
            # 把 chunk 转成普通 dict
            chunk_dict = chunk_to_dict(chunk, self.model)
            chunk_dict["bill"] = None
            chunk_dict["latency"] = None
            chunk_dict["routed_model"] = {"proposer": reference_models, "aggregator": agg_model}
            # yield JSON 字符串，SSE 格式
            yield f"data: {json.dumps(chunk_dict)}\n\n"
        end_time = time.time()
        if final_usage is not None:
            bill['prompt_tokens'][agg_model] += final_usage.prompt_tokens
            bill['completion_tokens'][agg_model] += final_usage.completion_tokens
        else:
            logger.warning("final_usage is None, skipping token billing for agg_model")
        for model in self.candidate_models:
            bill['cost'] += self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1000000 \
            + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1000000
        # 最后发送 summary chunk
        summary_chunk = {
            "id": None,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": self.model,
            "choices": [{
                "delta": {
                    "content": "",
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                    "refusal": None
                },
                "index": 0,
                "finish_reason": None,
                "logprobs": None
                }],
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": reference_models, "aggregator": agg_model}
        }
        yield f"data: {json.dumps(summary_chunk)}\n\n"


    async def process_func_moa(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        orig_system_prompt = None
        if messages[0]["role"] == "system":
            orig_system_prompt = messages[0]["content"]

        references = item.get("references", [])

        emoa_pipeline = self.emoa_pipeline
        # reference_models最多3个，如果大于3个会随机选择3个
        reference_models = self.candidate_models   #random.sample(self.candidate_models, 3)  
        agg_model = self.agg_model
        print(f"reference_models: {reference_models}")
        print(f"agg_model: {agg_model}") 
        
        if len(references) == 0 and len(reference_models) > 0:

            prev_references = []

            for i_round in range(self.rounds):
                if DEBUG:
                    logger.info(
                        f"Round {i_round+1}/{self.rounds} to collecting reference responses."
                    )

                async_responses = []
                for reference_model in reference_models:
                    async_response = emoa_pipeline.generate_with_references(
                        model=reference_model,
                        messages=messages,
                        references=prev_references,
                        orig_system_prompt=orig_system_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                        max_tokens=self.max_tokens,
                        api_info=self.api_info,
                        extra_headers=self.extra_headers
                    )
                    
                    async_responses.append(async_response)
                
                # Await all responses without blocking main process
                responses = await asyncio.gather(*async_responses)
                references = []
                for model, response in zip(reference_models, responses):
                    try:
                        ref = response.choices[0].message.content.strip()
                    except:
                        ref = ""
                        logger.warning(f"Failed to get response from model {model}")
                    if ref:
                        references.append(ref)
                        bill['prompt_tokens'][model] += response.usage.prompt_tokens
                        bill['completion_tokens'][model] += response.usage.completion_tokens

                if i_round < self.rounds - 1:
                    prev_references = references
                    references = []

                await asyncio.sleep(0.1)

        logger.info(f"{references}")

        agg_prompt = dedent("""
        You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

        Responses from models:
        """)
        response = await emoa_pipeline.generate_with_references(
            model=agg_model,
            messages=messages,
            references=references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=agg_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
            extra_headers=self.extra_headers
        )
        end_time = time.time()

        bill['prompt_tokens'][agg_model] += response.usage.prompt_tokens
        bill['completion_tokens'][agg_model] += response.usage.completion_tokens

        for model in self.candidate_models:
            bill['cost'] += self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1000000 \
            + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1000000

        return {
            "id": None,
            "object": "chat.completion",
            "created": time.time(),
            "model": self.model,
            "agg_model": agg_model,
            "choices": response.choices,
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": reference_models, "aggregator": agg_model}
        }


    async def process_func_moa_stream(
        self,
        item,
    ):
        bill = dict(
            prompt_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            completion_tokens=dict(
                zip(self.candidate_models, [0]*len(self.candidate_models))
            ), 
            cost=0
        )
        start_time = time.time()
        messages = item["messages"]
        orig_system_prompt = None
        if messages[0]["role"] == "system":
            orig_system_prompt = messages[0]["content"]

        references = item.get("references", [])

        emoa_pipeline = self.emoa_pipeline
        # reference_models最多3个，如果大于3个会随机选择3个
        reference_models = random.sample(self.candidate_models, 3)  
        agg_model = self.agg_model
        print(f"reference_models: {reference_models}")
        print(f"agg_model: {agg_model}") 
        
        if len(references) == 0 and len(reference_models) > 0:

            prev_references = []

            for i_round in range(self.rounds):
                if DEBUG:
                    logger.info(
                        f"Round {i_round+1}/{self.rounds} to collecting reference responses."
                    )

                async_responses = []
                for reference_model in reference_models:
                    async_response = emoa_pipeline.generate_with_references(
                        model=reference_model,
                        messages=messages,
                        references=prev_references,
                        orig_system_prompt=orig_system_prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout,
                        max_tokens=self.max_tokens,
                        api_info=self.api_info,
                        extra_headers=self.extra_headers
                    )
                    
                    async_responses.append(async_response)
                
                # Await all responses without blocking main process
                responses = await asyncio.gather(*async_responses)
                references = []
                for model, response in zip(reference_models, responses):
                    try:
                        ref = response.choices[0].message.content.strip()
                    except:
                        ref = ""
                        logger.warning(f"Failed to get response from model {model}")
                    if ref:
                        references.append(ref)
                        bill['prompt_tokens'][model] += response.usage.prompt_tokens
                        bill['completion_tokens'][model] += response.usage.completion_tokens

                if i_round < self.rounds - 1:
                    prev_references = references
                    references = []

                await asyncio.sleep(0.1)

        logger.info(f"{references}")

        agg_prompt = dedent("""
        You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

        Responses from models:
        """)
        

        # # 👇 新增：记录完整输出内容，用于估算 completion_tokens
        # full_response_text = ""

        async for chunk in emoa_pipeline.generate_with_references_stream(
            model=agg_model,
            messages=messages,
            references=references,
            orig_system_prompt=orig_system_prompt,
            system_prompt_to_inject=agg_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
            output_messages=True,
            api_info=self.api_info,
            extra_headers=self.extra_headers
        ):
            # content = chunk.choices[0].delta.content or ""
            # full_response_text += content

            if chunk.choices[0].finish_reason is not None:
                final_usage = chunk.usage
                break

            chunk_dict = chunk_to_dict(chunk, self.model)
            chunk_dict["bill"] = None
            chunk_dict["latency"] = None
            chunk_dict["routed_model"] = {"proposer": reference_models, "aggregator": agg_model}
            yield f"data: {json.dumps(chunk_dict)}\n\n"

        end_time = time.time()

        # 👇 更新账单
        if final_usage is not None:
            bill['prompt_tokens'][agg_model] += final_usage.prompt_tokens
            bill['completion_tokens'][agg_model] += final_usage.completion_tokens
        else:
            logger.warning("final_usage is None, skipping token billing for agg_model")
            # logger.warning("final_usage is None, estimating tokens with tiktoken")
            # # 初始化 tokenizer（只初始化一次）
            # breakpoint()
            # if not hasattr(self, '_est_tokenizer'):
            #     # 根据你的模型选择对应的 Qwen tokenizer
            #     model_name_or_path = "/mnt/gemininjceph2/geminicephfs/wx-dc-plt-hpc/mmdc-intern/jizewang/emoa/qwen_tokenizer"  # 👈 根据你实际模型改！
            #     self._est_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            # completion_token_estimate = len(self._est_tokenizer.encode(full_response_text))
            # # 👇 新增：估算 prompt_tokens —— 粗略拼接所有字符串内容
            # prompt_token_estimate = 0

            # # 加入 system prompt
            # if orig_system_prompt:
            #     prompt_token_estimate += len(self._est_tokenizer.encode(orig_system_prompt))

            # # 加入 messages
            # for msg in messages:
            #     content = msg["content"]
            #     if isinstance(content, list):
            #         text_parts = []
            #         for item in content:
            #             if isinstance(item, dict) and item.get("type") == "text":
            #                 text = item.get("text", "")  # 默认空字符串，避免 None 或缺失
            #                 if isinstance(text, str):
            #                     text_parts.append(text)
            #         content = "".join(text_parts)
            #     if isinstance(content, str):
            #         prompt_token_estimate += len(self._est_tokenizer.encode(content))

            # # 加入 references
            # for ref in references:
            #     if isinstance(ref, str):
            #         prompt_token_estimate += len(self._est_tokenizer.encode(ref))

            # # 加入 agg_prompt
            # if agg_prompt:
            #     prompt_token_estimate += len(self._est_tokenizer.encode(agg_prompt))

            # bill['prompt_tokens'][agg_model] += prompt_token_estimate
            # bill['completion_tokens'][agg_model] += completion_token_estimate

        for model in self.candidate_models:
            bill['cost'] += self.api_info[model]['input_price'] * bill['prompt_tokens'][model] / 1000000 \
            + self.api_info[model]['output_price'] * bill['completion_tokens'][model] / 1000000
        # 最后发送 summary chunk
        summary_chunk = {
            "id": None,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": self.model,
            "choices": [{
                "delta": {
                    "content": "",
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": None,
                    "refusal": None
                },
                "index": 0,
                "finish_reason": None,
                "logprobs": None
                }],
            "bill": bill,
            "latency": end_time - start_time,
            "routed_model": {"proposer": reference_models, "aggregator": agg_model}
        }
        yield f"data: {json.dumps(summary_chunk)}\n\n"


    def _get_process_function(self, model: str = "router"):
        return self.process_functions[model]

    def _get_process_function_stream(self, model: str = "router"):
        return self.process_functions_stream[model]

    @property
    def process_functions(self) -> Dict:
        return {
            "moa": self.process_func_moa,
            "routemoa": self.process_func_routemoa,
            "router": self.process_func_router
        }

    @property
    def process_functions_stream(self) -> Dict:
        return {
            "moa": self.process_func_moa_stream,
            "routemoa": self.process_func_routemoa_stream,
            "router": self.process_func_router_stream
        }


if __name__ == "__main__":
    Fire(EmoaApp)
