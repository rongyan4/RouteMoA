from openai import OpenAI, AsyncOpenAI
from loguru import logger
from typing import List, Optional, Dict, AsyncGenerator
import os
import time
from types import SimpleNamespace
import json
import asyncio

DEBUG = int(os.environ.get("DEBUG", "0"))

async def openai_api_generate(
        messages: List[Dict],
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        timeout: Optional[int] = 100,
        extra_headers: Optional[Dict] = None,
        stream: Optional[bool] = False,
        **kwargs,
    ):
    
    RETRY_SECONDS = [1]

    # Validate messages
    if not messages or not all(isinstance(msg, dict) and 'content' in msg for msg in messages):
        logger.error("Invalid messages format: %s", messages)
        raise ValueError("Messages must be a non-empty list of dictionaries with 'content' key.")

    if api_key in os.environ:
        api_key = os.getenv(api_key)
    client = AsyncOpenAI(
        api_key=api_key or "YOUR_API_KEY",
        base_url=api_base,
    )

    last_exception = None
    
    for sleep_time in RETRY_SECONDS:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to {model}: `{api_base}`."
                )

            # For non-streaming response
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=extra_headers,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            return response

        except Exception as e:
            logger.error(f"API REQUEST ERROR: {model} - {e}")
            logger.info(f"Retry in {sleep_time}s..{api_base}")
            last_exception = e
            time.sleep(sleep_time)
    
    # If all retries failed
    logger.warning(f"API request failed after {len(RETRY_SECONDS)} attempts. Returning empty response.")
    logger.warning(f"Last error: {str(last_exception)}")
    
    # Create mock response
    mock_message = SimpleNamespace()
    mock_message.content = ""
    mock_message.role = "assistant"
    
    mock_choice = SimpleNamespace()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "error"
    
    mock_response = SimpleNamespace()
    mock_response.choices = [mock_choice]
    mock_response.id = "error_" + str(int(time.time()))
    mock_response.created = int(time.time())
    mock_response.model = model or "unknown"
    mock_response.object = "chat.completion"
    mock_response.usage = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0
    )
    
    return mock_response

async def openai_api_generate_stream(
        messages: List[Dict],
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        timeout: Optional[int] = 100,
        extra_headers: Optional[Dict] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
    """Streaming version of the API call"""
    RETRY_SECONDS = [1, 2, 4]

    # Validate messages
    if not messages or not all(isinstance(msg, dict) and 'content' in msg for msg in messages):
        logger.error("Invalid messages format: %s", messages)
        raise ValueError("Messages must be a non-empty list of dictionaries with 'content' key.")

    if api_key in os.environ:
        api_key = os.getenv(api_key)
    client = AsyncOpenAI(
        api_key=api_key or "YOUR_API_KEY",
        base_url=api_base,
    )

    # 构造 mock completion chunk（SimpleNamespace）
    mock_delta = SimpleNamespace(role="assistant", content="")
    mock_choice = SimpleNamespace(
        index=0,
        delta=mock_delta,
        logprobs=None,
        finish_reason="error",
    )
    mock_response = SimpleNamespace(
        id=f"error_{int(time.time())}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model or "unknown",
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
    )


    last_exception = None
    
    for sleep_time in RETRY_SECONDS:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to {model}: `{api_base}`."
                )
            
            kwargs.pop("stream", None)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=extra_headers,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                max_tokens=max_tokens,
                stream=True,
                stream_options={
                    "include_usage": True
                },
                **kwargs,
            )
            stream = await response
            async for chunk in stream:
                yield chunk
            return


        except Exception as e:
            logger.error(f"API REQUEST ERROR: {model} - {e}")
            logger.info(f"Retry in {sleep_time}s..{api_base}")
            last_exception = e
            # 使用 asyncio.sleep 避免阻塞事件循环
            await asyncio.sleep(0.1)
    
    # ================================
    # 🔴 所有重试失败：发送 mock 响应作为最后一个 chunk
    # ================================
    logger.warning(f"API request failed after {len(RETRY_SECONDS)} attempts. Returning mock error response.")
    logger.warning(f"Last error: {str(last_exception)}")

    mock_delta = SimpleNamespace(role="assistant", content="")
    mock_choice = SimpleNamespace(
        index=0,
        delta=mock_delta,
        logprobs=None,
        finish_reason="error",
    )
    mock_response = SimpleNamespace(
        id=f"error_{int(time.time())}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model or "unknown",
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        ),
    )

    yield mock_response
