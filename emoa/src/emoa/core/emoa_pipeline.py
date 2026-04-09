import os
import json
import time
import requests
import openai
import copy
import asyncio

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

        # self.model = model
        # self.api_base = api_base,
        # self.api_key = api_key,
        # self.max_tokens = max_tokens,
        # self.temperature = temperature
        pass

    def generate_together(
            self,
            messages,
        ):

        output = None

        for sleep_time in [1, 2, 4, 8, 16, 32]:
            try:
                # endpoint = "https://api.together.xyz/v1/chat/completions"
                endpoint = "https://openrouter.ai/api/v1/chat/completions"

                if DEBUG:
                    logger.debug(
                        f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{self.model}`."
                    )

                res = requests.post(
                    endpoint,
                    json={
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "temperature": (self.temperature if self.temperature > 1e-4 else 0),
                        "messages": messages,
                    },
                    headers={
                        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                    },
                )
                if "error" in res.json():
                    logger.error(res.json())
                    if res.json()["error"]["type"] == "invalid_request_error":
                        logger.info("Input + output is longer than max_position_id.")
                        return None

                output = res.json()["choices"][0]["message"]["content"]

                break

            except Exception as e:
                logger.error(e)
                if DEBUG:
                    logger.debug(f"Msgs: `{messages}`")

                logger.info(f"Retry in {sleep_time}s..")
                time.sleep(sleep_time)

        if output is None:

            return output

        output = output.strip()

        if DEBUG:
            logger.debug(f"Output: `{output[:20]}...`.")

        return output

    async def generate_together_stream(
            self,
            messages,
        ):

        endpoint = "https://openrouter.ai/api/v1"
        client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"), base_url=endpoint
        )
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if self.temperature > 1e-4 else 0,
            max_tokens=self.max_tokens,
            stream=True,  # this time, we set stream=True
        )

        return response
    
    async def generate_with_references(
            self,
            messages: List,
            model: str,
            api_base: Optional[str] = None,
            api_key: Optional[str] = None,
            max_tokens: Optional[int] = 2048,
            temperature: Optional[float] = 0.7,
            references: Optional[List] = None,
            orig_system_prompt: Optional[str] = None,
            system_prompt_to_inject: Optional[str] = None,
            output_messages: Optional[bool] = False,
            api_info: Dict = None,
            **kwargs
        ):
        """ Generate responses with references

        Args:
            messages (List): _description_
            model (str): _description_
            api_base (Optional[str], optional): _description_. Defaults to None.
            api_key (Optional[str], optional): _description_. Defaults to None.
            max_tokens (Optional[int], optional): _description_. Defaults to 2048.
            temperature (Optional[float], optional): _description_. Defaults to 0.7.
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
        # logger.info(f"{messages=}")

        model_api_provider = ModelAPIProvider(api_info=api_info)
        

        response = await model_api_provider.generate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # try:
        #     # Wrap the generation call in a 15s timeout
        #     response = await asyncio.wait_for(
        #         model_api_provider.generate(
        #             model=model,
        #             messages=messages,
        #             temperature=temperature,
        #             max_tokens=max_tokens,
        #             **kwargs
        #         ),
        #         timeout=15
        #     )
        # except asyncio.TimeoutError:
        #     # If timeout, return an empty string
        #     return ""


        return response

        # output = response.choices[0].message.content.strip()
        # token_info['prompt_tokens'] += response.usage.prompt_tokens
        # token_info['completion_tokens'] += response.usage.completion_tokens

        # if output_messages:
        #     return messages + [{"role": "assistant", "content": output}], token_info
        # else:
        #     return output, token_info
        