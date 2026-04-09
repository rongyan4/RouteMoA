import os
import json
import time
import requests
import openai
import copy
from typing import Optional, List, Dict
from openai import OpenAI, AsyncOpenAI

from loguru import logger

from emoa.utils.inject_references_to_messages import inject_references_to_messages, inject_references_to_messages_wuhan
from emoa.serve.models import openai_api_generate, ModelAPIProvider
from transformers import AutoTokenizer


DEBUG = int(os.environ.get("DEBUG", "0"))


class EmoaPipeline:

    def __init__(
            self,
            streaming=False,
        ) -> None:
        pass
    
    async def generate_with_references(
            self,
            messages: List,
            model: str,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            max_tokens: Optional[int] = 131072,
            temperature: Optional[float] = 1,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            references: Optional[List] = None,
            orig_system_prompt: Optional[str] = None,
            system_prompt_to_inject: Optional[str] = None,
            output_messages: Optional[bool] = False,
            api_info: Dict = None,
            extra_headers: Dict = None,
            **kwargs
        ):
        """ Generate responses with references

        Args:
            messages (List): _description_
            model (str): _description_
            api_base (Optional[str], optional): _description_. Defaults to None.
            api_key (Optional[str], optional): _description_. Defaults to None.
            max_tokens (Optional[int], optional): _description_. Defaults to 131072.
            temperature (Optional[float], optional): _description_. Defaults to 1.
            references (Optional[List], optional): _description_. Defaults to None.
            orig_system_prompt (Optional[str], optional): _description_. Defaults to None.
            system_prompt_to_inject (Optional[str], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        token_info = dict(prompt_tokens=0, completion_tokens=0)
        if references is not None:
            if len(references) > 0:
                messages = inject_references_to_messages(
                    messages=messages,
                    references=references,
                    orig_system_prompt=orig_system_prompt,
                    system_prompt_to_inject=system_prompt_to_inject,
                )
        
        else:
            messages = inject_references_to_messages_wuhan(
                messages=messages,
                references=None,
                orig_system_prompt=orig_system_prompt,
                system_prompt_to_inject=system_prompt_to_inject,
            )

        model_api_provider = ModelAPIProvider(api_info=api_info)
        
        response = await model_api_provider.generate(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs
        )

        return response


    async def generate_with_references_stream(
            self,
            messages: List,
            model: str,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            max_tokens: Optional[int] = 131072,
            temperature: Optional[float] = 1,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            references: Optional[List] = None,
            orig_system_prompt: Optional[str] = None,
            system_prompt_to_inject: Optional[str] = None,
            output_messages: Optional[bool] = False,
            api_info: Dict = None,
            extra_headers: Dict = None,
            **kwargs
        ):
        """ Generate responses with references

        Args:
            messages (List): _description_
            model (str): _description_
            api_base (Optional[str], optional): _description_. Defaults to None.
            api_key (Optional[str], optional): _description_. Defaults to None.
            max_tokens (Optional[int], optional): _description_. Defaults to 131072.
            temperature (Optional[float], optional): _description_. Defaults to 1.
            references (Optional[List], optional): _description_. Defaults to None.
            orig_system_prompt (Optional[str], optional): _description_. Defaults to None.
            system_prompt_to_inject (Optional[str], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        token_info = dict(prompt_tokens=0, completion_tokens=0)
        if references is not None:
            if len(references) > 0:
                messages = inject_references_to_messages(
                    messages=messages,
                    references=references,
                    orig_system_prompt=orig_system_prompt,
                    system_prompt_to_inject=system_prompt_to_inject,
                )
        
        else:
            messages = inject_references_to_messages_wuhan(
                messages=messages,
                references=None,
                orig_system_prompt=orig_system_prompt,
                system_prompt_to_inject=system_prompt_to_inject,
            )

        model_api_provider = ModelAPIProvider(api_info=api_info)
        
        async for chunk in model_api_provider.generate_stream(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            **kwargs
        ):
            yield chunk
