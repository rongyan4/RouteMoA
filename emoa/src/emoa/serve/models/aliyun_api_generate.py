import os
import time
from typing import List, Optional, Dict
from http import HTTPStatus
from logging import Logger

import dashscope
from dashscope import Generation

from emoa.constants import ALIYUN_API_MODELS

from loguru import logger


def aliyun_api_generate(
    messages: List[Dict],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 0.7,
    workspace: Optional[str] = None,
    logger: Logger = logger,
):

    if model not in ALIYUN_API_MODELS:
        raise ValueError(f'{model} not available. Only the following models are currently supported: {ALIYUN_API_MODELS}')

    if isinstance(messages, str):
        raw_messages = [{'role': 'user', 'content': messages}]
    else:
        raw_messages = messages

    api_key = api_key or os.environ["ALIYUN_API_MODEL_KEY"]
    workspace = workspace or os.environ["ALIYUN_API_MODEL_WORKSPACE_ID"]

    # Request
    RETRY_SECONDS = [1, 2, 4]
    for sleep_time in RETRY_SECONDS:
        try:
            payload = dict(
                model=model,
                messages=raw_messages,
                api_key=api_key,
                workspace=workspace,
                temperature=temperature,
                stream=False,
                result_format="message",
            )
            payload.update({
                "enable_search": False,
            })

            # Make requests
            # logger.info(f"==== request ====\n{payload}")

            res = dashscope.Generation.call(**payload)

            if res.status_code != HTTPStatus.OK:
                full_err_msg = {
                    'model_name': model,
                    'request_id': res.request_id, 
                    'status_code': res.status_code, 
                    'error_code': res.code, 
                    'error_message': res.message,
                }
                logger.error(f"==== error ====\n{full_err_msg}")

                logger.info(f"Retry in {sleep_time}s..Aliyun Dashscope API")
                time.sleep(sleep_time)
                continue

            output = res.output.choices[0].message.content
            break
                
        except Exception as e:
            logger.error(f"API REQUEST ERROR: {model} - {e}")
            full_err_msg = {
                'model_name': model,
                'request_id': res.request_id, 
                'status_code': res.status_code, 
                'error_code': res.code, 
                'error_message': res.message,
            }
            logger.error(f"==== error ====\n{full_err_msg}")

            logger.info(f"Retry in {sleep_time}s..Aliyun Dashscope API")
            time.sleep(sleep_time)
    else:
        raise TimeoutError(f"API request unsuccessful after {sum(RETRY_SECONDS)}s.")

    output = output.strip()

    return output
