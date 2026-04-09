from pathlib import Path

REPO_ROOT_PATH = Path(__file__).parent.parent.parent
PACKAGE_ROOT_PATH = Path(__file__).parent

API_ENDPOINTS_JSON = REPO_ROOT_PATH / "configs/api_endpoints.json"

ALIYUN_API_MODELS = [
    "c4ai-command-r-plus",
    "deepseek-moe-16b-chat",
    "deepseek-llm-67b-chat",
    "deepseek-v2-chat",
    "dbrx-instruct",
    "internlm2-7b-chat",
    "internlm2-20b-chat",
    "mixtral-8x22b-instruct",
    "qwen1.5-7b-chat",
    "qwen1.5-14b-chat",
    "qwen1.5-32b-chat",
    "qwen1.5-72b-chat",
    "qwen2-72b-instruct",
    "llama3-8b-instruct",
    "llama3-70b-instruct",
    "yi-34b-chat",
    "yi-1.5-34b-chat",
]
