# LMDeploy Deployment Guide

## Single Model Startup Example
```bash
CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server \
  /home/1002/wangjize/hf_hub/Qwen2.5-Coder-7B-Instruct \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --server-port 10403 \
  --tp 1
```

## Cache and Environment Variables
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/home/1002/wangjize/hf_hub/
export MODELSCOPE_CACHE=/home/1002/wangjize/hf_hub/
```

## Service Endpoints
```bash
source /home/1003/miniconda3/etc/profile.d/conda.sh
conda activate wh_mmad
python3 -m emoa.serve.app_v2 --config emoa/configs/emoa_v2.json --host 0.0.0.0 --port 10666

# python -m emoa.serve.model_alias_proxy
```

## Model Download
```bash
export HF_HOME=/home/1002/wangjize/wuhan/RouteMoA/hf_hub
huggingface-cli download ContactDoctor/Bio-Medical-Llama-3-8B \
  --token [your-huggingface-token] \
  --local-dir /home/1002/wangjize/wuhan/RouteMoA/hf_hub/ContactDoctor/Bio-Medical-Llama-3-8B
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /home/1002/wangjize/wuhan/RouteMoA/hf_hub/Qwen/Qwen2.5-Coder-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir /home/1002/wangjize/wuhan/RouteMoA/hf_hub/Qwen/Qwen2.5-Math-7B-Instruct
huggingface-cli download google/gemma-2-9b-it \
  --token [your-huggingface-token] \
  --local-dir /home/1002/wangjize/wuhan/RouteMoA/hf_hub/google/gemma-2-9b-it
huggingface-cli download mistralai/Ministral-8B-Instruct-2410 \
  --local-dir /home/1002/wangjize/wuhan/RouteMoA/hf_hub/mistralai/Ministral-8B-Instruct-2410
```

## 5-GPU Deployment Script
```bash
mkdir -p logs
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nohup sh -c "CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server \
  /home/1002/wangjize/wuhan/RouteMoA/hf_hub/google/gemma-2-9b-it \
  --model-name google/gemma-2-9b-it \
  --server-port 10401 \
  --tp 1 \
  --model-format hf" > logs/gemma.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=1 lmdeploy serve api_server \
  /home/1002/wangjize/wuhan/RouteMoA/hf_hub/mistralai/Ministral-8B-Instruct-2410 \
  --model-name mistralai/Ministral-8B-Instruct-2410 \
  --server-port 10402 \
  --tp 1 \
  --model-format hf" > logs/ministral.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server \
  /home/1002/wangjize/wuhan/RouteMoA/hf_hub/Qwen/Qwen2.5-Coder-7B-Instruct \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --server-port 10403 \
  --tp 1 \
  --model-format hf" > logs/qwen-coder.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server \
  /home/1002/wangjize/wuhan/RouteMoA/hf_hub/Qwen/Qwen2.5-Math-7B-Instruct \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --server-port 10404 \
  --tp 1 \
  --model-format hf" > logs/qwen-math.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=4 lmdeploy serve api_server \
  /home/1002/wangjize/wuhan/RouteMoA/hf_hub/ContactDoctor/Bio-Medical-Llama-3-8B \
  --model-name ContactDoctor/Bio-Medical-Llama-3-8B \
  --server-port 10405 \
  --tp 1 \
  --model-format hf" > logs/biomedical.log 2>&1 &
```

