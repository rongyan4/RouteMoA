import json
import time
import os
import asyncio
from typing import List, Dict, Optional
from loguru import logger
import aiohttp
import anthropic

DEBUG = int(os.environ.get("DEBUG", "0"))

import os
import asyncio
import json
from typing import List, Dict, Optional
import aiohttp
from loguru import logger

DEBUG = int(os.environ.get("DEBUG", "0"))

async def claude_api_generate(
        messages: List[Dict],
        api_base: str = None,
        api_key: str = None, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.7,
        **kwargs,
    ):
    """
    Asynchronously call Claude API to generate text.

    Args:
        api_key (str): API key.
        model (str): Claude model name to use.
        messages (list): Message list containing role and content.
        max_tokens (int): Max tokens to generate, default is 2048.
        temperature (float): Controls randomness, default is 0.7.

    Returns:
        str: Generated text returned by Claude.
    """
    # Set max retry time
    RETRY_SECONDS = [1, 2, 4]

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Extract and process "system" entries
    system_prompt = []
    for entry in messages:
        if entry["role"] == "system":
            system_prompt.append(entry["content"])
    messages = [entry for entry in messages if entry["role"] != "system"]

    for sleep_time in RETRY_SECONDS:
        try:
            if DEBUG:
                logger.debug(f"Sending messages ({len(messages)}) to model: `{model}`.")
            
            # Synchronous call to Claude API
            message = client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=max_tokens,
                messages=messages,
            )

            # Return generated text
            return message.content[0].text.strip()

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            logger.info(f"Retrying in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)

    else:
        raise TimeoutError(f"API request unsuccessful after {sum(RETRY_SECONDS)} seconds.")        



async def claude_allesapin_generate(
        messages: List[Dict],
        api_base: str,
        api_key: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.7,
        **kwargs,
    ):
    """
    Asynchronously call Claude AllesAPIN API to generate results.

    Args:
        messages (List[Dict]): Message list, each message is a dict containing role and content.
        api_base (str): API URL。
        api_key (str): API key.
        model (str, optional): Model name, default uses the provided model.
        max_tokens (int, optional): Max tokens, default is 512.
        retry (int, optional): Number of retries, default is 2.

    Returns:
        str: Generated text.
    """
    # Set max retry time
    RETRY_SECONDS = [1, 2, 4]

    # Prepare request headers
    if api_key in os.environ:
        api_key = os.getenv(api_key)
    headers = {
        'alles-apin-token': api_key,
        'content-type': 'application/json',
    }
    # Initialize variable to store "system" entry content
    system_prompt = None

    # Find and remove entry with role "system", and record its content
    messages = [entry for entry in messages if entry["role"] != "system" or (system_prompt := entry["content"]) is None]
    
    # Set basic data for API request
    data = {
        'model': model or "default-model",  # Default model is "default-model"
        'system': system_prompt,
        'messages': messages,
        'max_tokens': max_tokens,
    }

    # Initialize client
    async with aiohttp.ClientSession() as session:
        for sleep_time in RETRY_SECONDS:
            try:
                if DEBUG:
                    logger.debug(
                        f"Sending messages ({len(messages)}) to model: `{model}` at {api_base}."
                    )

                # Send POST request
                async with session.post(api_base, headers=headers, json=data) as response:
                    raw_response = await response.text()
                    try:
                        response_data = json.loads(raw_response)
                    except json.JSONDecodeError:
                        if 'https://errors.aliyun.com/images' in raw_response:
                            return 'request blocked by allesapin'
                        logger.error(f'JsonDecode error, got: {raw_response}')
                        continue

                    # Process response
                    if response.status == 200 and response_data.get('msgCode') == '10000':
                        generated_text = response_data['data']['content'][0]['text'].strip() 
                        breakpoint()
                        logger.debug(f"Generated: {generated_text}")
                        return generated_text
                    else:
                        logger.error(f"Error: {response_data.get('data')}")
                        continue
            except aiohttp.ClientError as e:
                logger.error(f"API REQUEST ERROR: {e}")
                logger.info(f"Retrying in {sleep_time}s..")
                await asyncio.sleep(sleep_time)  # Async wait
        else:
            raise TimeoutError(f"API request unsuccessful after {sum(RETRY_SECONDS)}s.")

