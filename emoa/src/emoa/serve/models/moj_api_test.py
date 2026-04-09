from openai import OpenAI
import json


client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://10.140.0.131:10011/v1"
)
# model_name = ""
# model_name = client.models.list().data[0].id
# model_name = "internlm2-chat-20b"
# model_name = "internlm2_5-7b-chat"
# model_name = "glm-4-9b-chat"

stream = False

payload = {
    "model": "emoa_mixed_models",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what model are you?"}
    ],
    "max_tokens": 2048,
    "temperature": 0,
    "extra_body": {
        # "intermediate_prompt": "If you believe Assistant A performed better, please output A and provide your reasoning.\nIf you believe Assistant B performed better, please output B and provide your reasoning.\nPlease output:",
        # "final_output_prompt": "If you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option.\nPlease output:"
    }
}

# {
#     "model": "emoa_mixed_models",
#     "messages": [
#         {"role": "user", "content": "Please read the dialogue between the two assistants and the user to determine which assistant performed better during the conversation.\nHere is the dialogue content:\n[Dialogue Begin]\n### User: Load the gsm8k dataset and return a list of dictionaries containing questions and answers.\n[Dialogue End]\nIf you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option.\nPlease output:"}
#     ],
#     "max_tokens": 2048,
#     "temperature": 0,
#     "extra_body": {
#         "rounds": 2,
#         "agg_model": "internlm2_5-7b-chat",
# "reference_models": [
#         "qwen2-7b-instruct",
#         "glm-4-9b-chat",
#         "llama-3.1-8b-instruct",
#         "internlm2_5-7b-chat",
#         "yi-1.5-9b-chat"
#     ],
#         "intermediate_prompt": "If you believe Assistant A performed better, please output A and provide your reasoning.\nIf you believe Assistant B performed better, please output B and provide your reasoning.\nPlease output:",
#         "final_output_prompt": "If you believe Assistant A performed better, please output A directly.\nIf you believe Assistant B performed better, please output B directly.\nDo not output any other content, just the option.\nPlease output:" 
#     }
# }

# print(json.dumps(payload))

response = client.chat.completions.create(
    **payload,
    # top_p=0.8
)

print(response)

# if stream:
#     for chunk in response:
#         print(chunk.choices[0].delta.content, end = "", flush=True)
# else:
#     print(response.choices[0].message.content)
