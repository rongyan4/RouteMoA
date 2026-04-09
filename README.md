# RouteMoA

English | [简体中文](README_cn.md)

**[ACL 2026] RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents**

Welcome to the official repository for **RouteMoA**. This project introduces an efficient and dynamic routing mechanism designed to boost the performance of Mixture-of-Agents (MoA) architectures without relying on pre-inference steps.

---

## Table of Contents

- [RouteMoA](#routemoa)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [1. Deploy the LLMs](#1-deploy-the-llms)
    - [2. Start the RouteMoA Service](#2-start-the-routemoa-service)
    - [3. Evaluate the Service](#3-evaluate-the-service)
  - [Router Training](#router-training)
  - [Acknowledgements](#acknowledgements)

---

## Installation

We highly recommend using `conda` to create an isolated environment. 

> **System Requirement:** We strongly recommend using a Linux (x86_64) operating system. This guide is based on Ubuntu 22.04, and other operating systems are currently not officially supported.

```bash
conda create -n YOUR_ENV_NAME python=3.10 -y
conda activate YOUR_ENV_NAME

# Install the OpenCompass evaluation framework
cd opencompass 
pip install -e .

# Install the EMoA core module
cd ../emoa
pip install -e .
```

---

## Quick Start

### 1. Deploy the LLMs

We suggest downloading the LLM checkpoints locally first. Our experiments were conducted using five NVIDIA A800 80GB GPUs. For reproducibility, **We strongly recommend deploying each model on a dedicated high-performance NVIDIA GPU with more than 70GB of VRAM.**

Set up your local checkpoint directory:
```bash
mkdir -p </path/to/your/local/checkpoint/folder>
```

Install the HuggingFace CLI tool:
```bash
pip install -U huggingface_hub
```

Download the required models from HuggingFace:
```bash
# Download Bio-Medical Llama
huggingface-cli download ContactDoctor/Bio-Medical-Llama-3-8B \
  --token <your-huggingface-token> \
  --local-dir </path/to/your/local/checkpoint/folder>/Bio-Medical-Llama-3-8B

# Download Qwen models
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir </path/to/your/local/checkpoint/folder>/Qwen2.5-Coder-7B-Instruct

huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir </path/to/your/local/checkpoint/folder>/Qwen2.5-Math-7B-Instruct 

# Download Gemma and Ministral models
huggingface-cli download google/gemma-2-9b-it \
  --token <your-huggingface-token> \
  --local-dir </path/to/your/local/checkpoint/folder>/gemma-2-9b-it

huggingface-cli download mistralai/Ministral-8B-Instruct-2410 \
  --local-dir </path/to/your/local/checkpoint/folder>/Ministral-8B-Instruct-2410
```

Once the models are ready, deploy them using LMDeploy:
```bash
bash lmdeploy.sh
```

### 2. Start the RouteMoA Service

Next, set up the router and launch the core services.

1. **Download the Router checkpoint** from [Google Drive](https://drive.google.com/file/d/1cMntfcUJ6mKf5a2bsgwwWo4ehCLWqAuw/view?usp=sharing).
2. Place the downloaded checkpoint into your local `checkpoints/` directory.
3. **Download the router backbone** (`mdeberta-v3-base`) from the official Microsoft project:
   - Model weights: [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
   - Or simply use the CLI:
   ```bash
   huggingface-cli download microsoft/mdeberta-v3-base --local-dir </path/to/mdeberta-v3-base>
   ```
4. **Update the config file** (`emoa/configs/emoa_v2.json`):
   - Set `router_pth_path` to the absolute path of your downloaded Router checkpoint.
   - Set `router_backbone` to the absolute path of your `mdeberta-v3-base` folder.

Now, start the services:
```bash
conda activate YOUR_ENV_NAME

# Start the EMoA service
python3 -m emoa.serve.app_v2 --config emoa/configs/emoa_v2.json --host 0.0.0.0 --port 10666

# Start the SMoA service
python3 -m emoa.serve.smoa --config emoa/configs/smoa.json --host 0.0.0.0 --port 10667
```

### 3. Evaluate the Service

We use OpenCompass to evaluate the performance of our EMoA service. 

First, run the inference:
```bash
conda activate YOUR_ENV_NAME
opencompass examples/eval_emoa.py -r latest --mode infer --dump-eval-details
```

After inference completes, run the evaluation to calculate the scores:
```bash
opencompass examples/eval_emoa.py -r latest --mode eval --dump-eval-details
```

If you need to analyze the API costs and latency, we provide a handy script:
```bash
cd opencompass
python bill_stat.py
```

> **Note on Reproducibility:** To demonstrate the reproducibility of our experiments, we have provided the full OpenCompass evaluation results. You can download and verify them [here](https://drive.google.com/file/d/1QAIy1lxqvXrPFoQj0g6tjHXMWxowH0Je/view?usp=sharing).

---

## Router Training

Our router training pipeline is built on top of [RouterDC](https://github.com/shuhao02/RouterDC). 

You can train your own router using custom data by following the instructions provided in the RouterDC repository. We have currently released the pre-trained checkpoint of our router, and the full training code tailored for RouteMoA will be released in the future.

---

## Acknowledgements

This project is built upon the great work from the open-source community. We sincerely thank:
- [OpenCompass](https://github.com/open-compass/opencompass)
- [RouterDC](https://github.com/shuhao02/RouterDC)
