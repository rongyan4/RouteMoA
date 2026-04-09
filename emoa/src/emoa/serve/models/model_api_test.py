from openai import OpenAI


client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://10.140.0.131:10012/v1"
)
# model_name = client.models.list().data[0].id
model_name = "internlm2-chat-20b"
# model_name = "internlm2_5-7b-chat"

stream = False

payload = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what model are you?"}
    ],
    "max_tokens": 512,
    "temperature": 0.1,
    "stream": False
}

response = client.chat.completions.create(
    **payload,
    # top_p=0.8
)

print(response)

if stream:
    for chunk in response:
        print(chunk.choices[0].delta.content, end = "", flush=True)
else:
    print(response.choices[0].message.content)