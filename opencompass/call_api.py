from opencompass.models.openai_api import OpenAI, OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])

api_model = OpenAISDK(
    abbr='emoa-oracle',
    path='moa_mixed_models',
    openai_api_base='http://127.0.0.1:10074/v1',
    key='YOUR_API_KEY',
    retry=10,
    meta_template=api_meta_template,
    rpm_verbose=True,
    query_per_second=1,  # 0.05 or 1
    max_out_len=8192,
    max_seq_len=131072,
    batch_size=5,  # 1 or 16
    temperature=0.7,
    verbose=True,
    tokenizer_path='Qwen/Qwen2.5-72B-Instruct',
    timeout=2000
)

api_model.generate()