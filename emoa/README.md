## Installation
```bash
# Create a conda virtual environment
conda create -n Emoa python=3.10

conda activate Emoa

# Automatially installs project and all dependencies
pip install -e .
```

## Usage

This MoA/Emoa API server currently only supports making API calls to models for inference, so you would need to either:

- Serve your own LLMs and provide a base URL and API key in the config file; or

- Use a commercially available API model (with high RPM limits)


### Getting your models ready for inference

Skip this section if you already have API models available. Only the following API types are supported:

1. `openai_api_generate`: OpenAI-compatible API models.

2. `aliyun_api_generate`: Alibaba Cloud-compatible API models.

You can add more support to `src/emoa/serve/models/model_api_provider.py`

Alternatively, you can serve your own LLM for the purpose of MoA/Emoa inference (as reference models, aggregation models, or both). Here is an example of serving an internlm2.5-7b-chat model on S1/S2 cluster using `lmdeploy`:

On S1/S2 cluster:
```bash
# Request and enter a compute node. Choose the appropriate number of gpus
srun \
    -p llmeval \
    --quotatype=auto \
    --job-name=emoa_model_server \
    --gres=gpu:2 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
    --pty \
    bash -l serve_local_llm.sh
```

where `serve_local_llm.sh` contains the following code:

```bash
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=<path/to/cache/dir>
export HUGGINGFACE_HUB_CACHE=<path/to/cache/dir>
export TRANSFORMERS_CACHE=<path/to/cache/dir>
conda activate Emoa
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

lmdeploy serve api_server internlm/internlm2_5-7b-chat \
    --model-name internlm2_5-7b-chat \
    --tp 2 \
    --log-level INFO \
    --server-port 10001
```

You can test your API at the address: `https://<host>:10001`, where `<host>` is the ip address of your host machine.


Add your API model settings to the config file `configs/api_endpoints.json` with the following parameters:

1. `model_name`: Your API model name

2. `api_type`: The adapter/generator type used by your API model. See `src/emoa/serve/models/model_api_provider.py` for all supported `api_type`s.

3. `api_base`: The base URL of your API model.

4. `api_key`: The API Key for your API model. Note that if you don't set the `api_key`, the model generator will look for corresponding envrionment variable set by the `api_key_env` variable in `model_api_provider.py`.

The config file should have the following format:
```json
{
    "qwen2-7b-instruct": {
        "model_name": "qwen2-7b-instruct",
        "api_type": "openai",
        "api_base": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY"
    },
    "internlm2_5-7b-chat": {
        "model_name": "internlm2_5-7b-chat",
        "api_type": "openai",
        "api_base": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY"
    },
}
```

### Serving the MoA/Emoa API

Create a JSON config file in the `configs` directory to initialize the default settings for your Emoa API. For example, the following config file can be created:

`configs/agent_config.json`:
```json
{
    "model": "moa_mixed_models",
    "agg_model": "internlm2_5-7b-chat",
    "reference_models": [
        "qwen2-7b-instruct",
        "glm-4-9b-chat",
        "internlm2_5-7b-chat",
    ],
    "mode": "standard_moa",
    "tempurature": 0.7,
    "max_tokens": 1024,
    "rounds": 2,
    "output_path": "outputs/example.jsonl",
    "stream": false
}
```

1. `model`: The MoA/Emoa API's model name

2. `agg_model`: The aggregation model used to aggregate the final round of responses.

3. `reference_models`: The reference models in each round used to synthesize the responses from the previous round.

4. `mode`: The MoA/Emoa mode to use. Currently, there are 3 available modes:

    1. `standard_moa`: The standard mixture-of-agents configuration where in each round, all reference models synthesizes all responses from the previous round to the user's query, and a final aggregation model synthesizes the outputs from the last round to produce a single response.

    2. `standard_emoa`: Like `standard_moa` with the system prompt altered for LLM evaluation tasks. The models in each round are now encouraged to output reasoning along with their answers. The last aggregator takes previous rounds' outputs into consideration before producing a single response to the user's query.

    3. `mixed_temps`: Same as `standard_moa` except that in each round, each reference model is set a different tempurature to encourage varying responses. This is useful to increase response variation if your reference model count is low.

5. `tempurature`: Sets the tempurature for ALL models including reference and aggregation models.

6. `max_tokens`: Sets the max_tokens for ALL models including reference and aggregation models.

7. `rounds`: The number of rounds to synthesize model outputs by reference models before aggregation.

8. `output_path`: An optional output path to save model outputs when the API is being called. Only the final API response will be saved.

9. `stream`: Whether to turn on API streaming. Streaming is currently not supported, so this parameter is set to `false`.


Start the MoA/Emoa server by running the following commands:

```bash
# Export any API keys to environment variables
source scripts/set_api_keys.sh

# Turn off proxy if you're running on S1/S2 clusters
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

# Run the API server on current host and port 10013 that 
# can be accessed from any ip addresses
python3 -m emoa.serve.app \
    --config="configs/agent_config.json" \
    --host="0.0.0.0" \
    --port="10013"
```

You can test your API at the address: `https://<host>:10010`, where `<host>` is the ip address of your host machine.

### Making API calls for inference

Once you'd served and tested your MoA/Emoa server, you can make API calls to for model inference. The API is OpenAI-compatible so you can make model inference using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key='YOUR_API_KEY',  # API key not needed
    base_url="http://10.140.0.131:10010/v1"
)

payload = {
    "model": "moa_mixed_models",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what model are you?"}
    ],
    "max_tokens": 1024,  # Sets the max_tokens for all models
    "temperature": 0.7,  # Sets the tempurature for all models
    "extra_body": {
        "agg_model": "internlm2_5-7b-chat",
        "reference_models": [
            "qwen2-7b-instruct",
            "glm-4-9b-chat",
            "internlm2_5-7b-chat",
            "yi-1.5-9b-chat"
        ],
        "rounds": 2,
        "mode": "standard_moa"
    }
}

response = client.chat.completions.create(
    **payload,
)

print(response.choices[0].message.content)
```

**Warning**: 

1. Default config settings will be used if any optional parameters are not provided.

2. Any MoA-specific parameters are passed via the `extra_body` parameter.

