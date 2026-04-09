async def process_func_moa(
    self,
    item,
):
    # messages = [{"role": "user", "content": item[self.message_key]}]
    messages = item["messages"]
    orig_system_prompt = None
    if messages[0]["role"] == "system":
        orig_system_prompt = messages[0]["content"]

    references = item.get("references", [])

    emoa_pipeline = self.emoa_pipeline
    reference_models = self.candidate_models
    agg_model = self.agg_model
    if len(references) == 0 and len(reference_models) > 0:

        prev_references = []

        for i_round in range(self.rounds):
            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{self.rounds} to collecting reference responses."
                )

            responses = []
            for reference_model in reference_models:
                response = emoa_pipeline.generate_with_references(
                    # api_base=self.model_api_urls[reference_model],
                    # model=None,
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    orig_system_prompt=orig_system_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_info=self.api_info
                )
                
                responses.append(response)

            # Await all responses without blocking main process
            references = await asyncio.gather(*responses)

            references = [ref for ref in references if ref is not None]

            if i_round < self.rounds - 1:
                prev_references = references
                references = []

            time.sleep(0.1)

    logger.info(f"{references}")

    agg_prompt = dedent("""
    You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

    Responses from models:
    """)
    messages = await emoa_pipeline.generate_with_references(
        # api_base=self.model_api_urls[self.model],
        # model=None,
        model=agg_model,
        messages=messages,
        references=references,
        orig_system_prompt=orig_system_prompt,
        system_prompt_to_inject=agg_prompt,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        output_messages=True,
        api_info=self.api_info
    )

    # return {"output": output}
    # return {"output": output, "generator": self.model + "-together"}
    return {
        "id": None,
        "object": "chat.completion",
        "created": time.time(),
        "model": self.model,
        "agg_model": self.agg_model,
        "choices": [{
            "message": messages[-1]
        }],
        "messages_last_round": messages
    }