import copy
from textwrap import dedent
from typing import List, Optional
from loguru import logger


def inject_references_to_messages(
    messages: List,
    references: List,
    orig_system_prompt: Optional[str] = None,
    system_prompt_to_inject: Optional[str] = None,
):
    messages = copy.deepcopy(messages)

    if system_prompt_to_inject is None:
        system = dedent("""
        You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

        Responses from models:
        """)
    else:
        system = system_prompt_to_inject

    for i, reference in enumerate(references):
        system += f"\n[Reference {i+1} Start]\n{reference}\n[Reference {i+1} End]"

    if messages[0]["role"] == "system":
        if orig_system_prompt is None:
            messages[0]["content"] += "\n\n" + system
        else:
            messages[0]["content"] = orig_system_prompt + "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    # logger.info(f"{messages}")
    return messages



def inject_references_to_messages_wuhan(
    messages: List,
    references: None,
    orig_system_prompt: Optional[str] = None,
    system_prompt_to_inject: Optional[str] = None,
):
    messages = copy.deepcopy(messages)

    if system_prompt_to_inject is None:
        system = dedent("""Your are a good assistant for the user
        """)
    else:
        system = system_prompt_to_inject



    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
        
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages
