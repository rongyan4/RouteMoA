import os
import json
from typing import List, Dict, Optional, AsyncGenerator
from functools import partial

from emoa.constants import (
    API_ENDPOINTS_JSON    
)
from emoa.serve.models import (
    openai_api_generate,
    openai_api_generate_stream,
    claude_api_generate,
    claude_allesapin_generate
)


GENERATOR = {
    "openai": openai_api_generate,
    "claude_allesapin": claude_allesapin_generate,
    "claude": claude_api_generate
}

STREAM_GENERATOR = {
    "openai": openai_api_generate_stream,
    # Add other stream generators as needed
}


class ModelAPIProvider:

    def __init__(
            self,
            api_info: Dict = None,
        ) -> None:
        """ Model API Provider class for model response generation
        """
        self.api_info = api_info
    
    
    async def generate(
            self, 
            messages: List[Dict],
            model: Optional[str] = None,
            max_tokens: Optional[int] = 2048,
            temperature: Optional[float] = 0.7,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            extra_headers: Optional[Dict] = None,
            **kwargs,
        ):
        """ Generate response from specified API models
        """
        if model is None:
            raise ValueError("model not specified.")
        model_name = self.api_info[model]['model_name']
        api_base = self.api_info[model]['api_base']
        api_key = self.api_info[model]['api_key'] or 'YOUR_API_KEY'
        api_type = self.api_info[model]['api_type']
        
        if api_type not in GENERATOR:
            NotImplementedError(f"api_type: {api_type} not implemented.")
        api_generator = GENERATOR[api_type]

        if api_base is None:
            raise ValueError("api_base not specified.")
        
        output = await api_generator(
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            extra_headers=extra_headers,
            **kwargs,
        )
        return output


    async def generate_stream(
            self, 
            messages: List[Dict],
            model: Optional[str] = None,
            max_tokens: Optional[int] = 2048,
            temperature: Optional[float] = 0.7,
            top_p: Optional[float] = 0.9,
            timeout: Optional[int] = 100,
            extra_headers: Optional[Dict] = None,
            **kwargs,
        ):
        """ Generate response from specified API models
        """
        if model is None:
            raise ValueError("model not specified.")
        model_name = self.api_info[model]['model_name']
        api_base = self.api_info[model]['api_base']
        api_key = self.api_info[model]['api_key'] or 'YOUR_API_KEY'
        api_type = self.api_info[model]['api_type']

        if api_type not in STREAM_GENERATOR:
            NotImplementedError(f"Streaming not implemented for api_type: {api_type}")
        api_generator = STREAM_GENERATOR[api_type]

        if api_base is None:
            raise ValueError("api_base not specified.")
        
        async for chunk in api_generator(
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            extra_headers=extra_headers,
            **kwargs,
        ):
            yield chunk
