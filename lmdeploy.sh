set -e

mkdir -p logs

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOCAL_MODEL_DIR="your-local-llm-checkpoint-folder"

nohup sh -c "CUDA_VISIBLE_DEVICES=0 python -m lmdeploy serve api_server \
  ${LOCAL_MODEL_DIR}/Qwen/Qwen2.5-Coder-7B-Instruct \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --server-port 10403 \
  --tp 1" > logs/qwen-coder.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=1 python -m lmdeploy serve api_server \
  ${LOCAL_MODEL_DIR}/google/gemma-2-9b-it \
  --model-name google/gemma-2-9b-it \
  --server-port 10401 \
  --tp 1" > logs/gemma.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=2 python -m lmdeploy serve api_server \
  ${LOCAL_MODEL_DIR}/mistralai/Ministral-8B-Instruct-2410 \
  --model-name mistralai/Ministral-8B-Instruct-2410 \
  --server-port 10402 \
  --tp 1" > logs/ministral.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=3 python -m lmdeploy serve api_server \
  ${LOCAL_MODEL_DIR}/Qwen/Qwen2.5-Math-7B-Instruct \
  --model-name Qwen/Qwen2.5-Math-7B-Instruct \
  --server-port 10404 \
  --tp 1" > logs/qwen-math.log 2>&1 &
nohup sh -c "CUDA_VISIBLE_DEVICES=5 python -m lmdeploy serve api_server \
  ${LOCAL_MODEL_DIR}/ContactDoctor/Bio-Medical-Llama-3-8B \
  --model-name ContactDoctor/Bio-Medical-Llama-3-8B \
  --server-port 10405 \
  --tp 1" > logs/biomedical.log 2>&1 &
