from openai import OpenAI, AsyncOpenAI
from loguru import logger
from typing import List, Optional, Dict
import os
import time

DEBUG = int(os.environ.get("DEBUG", "0"))


def _normalize_messages_for_gemma(messages: List[Dict]) -> List[Dict]:
    system_chunks = [
        msg.get("content")
        for msg in messages
        if msg.get("role") == "system" and msg.get("content")
    ]
    filtered = [msg for msg in messages if msg.get("role") != "system"]
    if system_chunks:
        system_text = "\n".join(system_chunks).strip()
        if system_text:
            if filtered and filtered[0].get("role") == "user":
                merged = f"{system_text}\n\n{filtered[0].get('content', '')}".strip()
                filtered[0] = {**filtered[0], "content": merged}
            else:
                filtered.insert(0, {"role": "user", "content": system_text})
    return filtered


async def openai_api_generate(
        messages: List[Dict],
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.7,
        **kwargs,
    ):
    # client = OpenAI(
    #     api_key="EMPTY" if api_key == "" else api_key,
    #     base_url=api_base,
    # )
    # RETRY_SECONDS = [1, 2, 4, 8, 16, 32]
    RETRY_SECONDS = [1, 2, 4]

    if model and "gemma" in model.lower():
        messages = _normalize_messages_for_gemma(messages)

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

    for sleep_time in RETRY_SECONDS:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to {model}: `{api_base}`."
                )
            model_cards = await client.models.list()._get_page()

            if not model_cards or not hasattr(model_cards, "data") or not model_cards.data:
                logger.error("Failed to fetch model cards: %s", model_cards)
                raise ValueError("No valid models available from the API.")

            model = model or model_cards.data[0].id
            response = await client.chat.completions.create(
                model=model,
                # model="internlm2-chat-20b",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=2000,
            )
            if not response or not hasattr(response, "choices") or not response.choices:
                breakpoint()
                logger.error("Invalid API response: %s", response)
                raise ValueError("API response does not contain valid choices.")
            # output = response.choices[0].message.content
            # finish_reason = response.choices[0].finish_reason
            break

        except Exception as e:
            logger.error(f"API REQUEST ERROR: {model} - {e}")
            # breakpoint()
            logger.info(f"Retry in {sleep_time}s..{api_base}")
            time.sleep(sleep_time)
    else:
        raise TimeoutError(f"API request unsuccessful after {sum(RETRY_SECONDS)}s.")

    # output = output.strip()

    # return output
    return response
    
