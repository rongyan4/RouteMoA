from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate


"""
                infer   eval
Single Model    ok      ok
Routing     
    Oracle
    DC
    MoA         ok      ok
    sMoA
RouteMoA
    Oracle
    Vanilla
    v2
"""

from copy import deepcopy
from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.models.openai_api import OpenAI, OpenAISDK

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

#######################################################################
#                 PART 1  Dataset/Model Configuration                 #
#######################################################################

with read_base():
    # from opencompass.configs.datasets.math.math_emoa import math_datasets
    # from opencompass.configs.datasets.gsm8k.gsm8k_emoa import gsm8k_datasets
    # from opencompass.configs.datasets.ARC_c.ARC_c_emoa import ARC_c_datasets
    # from opencompass.configs.datasets.mmlu.mmlu_emoa import mmlu_datasets
    # # from opencompass.configs.datasets.ceval.ceval_emoa import emoa_ceval_datasets
    # from opencompass.configs.datasets.mbpp.mbpp_emoa import sanitized_mbpp_datasets
    # from opencompass.configs.datasets.race.race_emoa import race_datasets

    from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_6e39a4 import gsm8k_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_cot_gen_926652 import ARC_c_datasets
    from opencompass.configs.datasets.mmlu.mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets
    # from opencompass.configs.datasets.ceval.ceval_zero_shot_gen_bd40ef import ceval_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets
    from opencompass.configs.datasets.race.race_cot_gen_d95929 import race_datasets
    # from opencompass.configs.datasets.sym.sym_emoa import sym_datasets

    # from opencompass.configs.summarizers.groups.mmlu import mmlu_summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = []
model_configs = [
# # (abbr, path, num_gpus)
# # ('Mistral-Large-Instruct-2411', 'mistralai/Mistral-Large-Instruct-2411', 4),
# # ('Llama-3.1-Nemotron-70B-Instruct-HF', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', 4),
# # # ('DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 4),
# # ('Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 1),
# # # ('llama-3-chinese-8b-instruct-v3', 'hfl/llama-3-chinese-8b-instruct-v3', 1),

# ('gemma-2-9b-it', 'google/gemma-2-9b-it', 1),
# ('Ministral-8B-Instruct-2410', 'mistralai/Ministral-8B-Instruct-2410', 1),
# ('Qwen2.5-Coder-7B-Instruct', 'Qwen/Qwen2.5-Coder-7B-Instruct', 1),
# ('Qwen2.5-Math-7B-Instruct', 'Qwen/Qwen2.5-Math-7B-Instruct', 1),
# ('Bio-Medical-Llama-3-8B', 'ContactDoctor/Bio-Medical-Llama-3-8B', 1)
]


max_seq_len = 131072
max_out_len = 8192
max_batch_size = 128

for abbr, path, num_gpus in model_configs:
    if abbr is None:
        abbr = path.split('/')[-2] + '--' + path.split('/')[-1]

    model = dict(
        type=TurboMindModelwithChatTemplate,
        abbr=abbr,
        path=path,
        engine_config=dict(session_len=max_seq_len,
                           max_batch_size=max_batch_size,
                           tp=num_gpus),
        gen_config=dict(top_k=10,
                        temperature=0.7,
                        top_p=0.9,
                        max_new_tokens=max_out_len,
                        do_sample=True),
                        # output_logits='generation'),
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=max_batch_size,
        run_cfg=dict(num_gpus=num_gpus),
    )

    models.append(model)


#*********** for openai API-compitable *************
api_model_configs = [
    # (   'moa', # abbr
    #     "http://127.0.0.1:10077/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # ),

    (   'smoa-v1', # abbr
        "http://127.0.0.1:10088/v1", # base_url
        "YOUR_API_KEY", # key
        "moa_mixed_models", # model_name
        0.7, # temperature
    ),

    # (   'emoa-oracle', # abbr
    #     "http://127.0.0.1:10074/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # ),

    # (   'emoa-routerdc-v3', # abbr v2 uses 4 models with ministral as summarizer; v3 uses 4 models with the second highest scored model by router as summarizer
    #     "http://127.0.0.1:10078/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'emoa-routerdc-math', 
    #     "http://127.0.0.1:10078/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'emoa-routerdc-arc-v1',   # First layer and second layer both have 3 models
    #     "http://127.0.0.1:10079/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'emoa-routerdc-arc-v2',  #First layer 2 models, second layer 3 models
    #     "http://127.0.0.1:10068/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'emoa-routerdc-arc-v3',  #Use original router, push 2 rounds with threshold 0.82
    #     "http://127.0.0.1:10078/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'dylan-gr',  
    #     "http://127.0.0.1:10099/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    # (   'dylan-ar',  
    #     "http://127.0.0.1:10099/v1", # base_url
    #     "YOUR_API_KEY", # key
    #     "moa_mixed_models", # model_name
    #     0.7, # temperature
    # )

    

]

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
])


for abbr, base_url, key, model_name, temperature in api_model_configs:
    api_model = dict(type=OpenAISDK,
                     abbr=abbr,
                     path=model_name,
                     openai_api_base=base_url,
                     key=key,
                     retry=10,
                     meta_template=api_meta_template,
                     rpm_verbose=True,
                     query_per_second=0.25,  # 0.05 or 1
                     max_out_len=8192,
                     max_seq_len=131072,
                     batch_size=1,  # 1 or 16
                     temperature=temperature,
                     verbose=True,
                     tokenizer_path='Qwen/Qwen2.5-72B-Instruct',
                     timeout=2000)
    models.append(api_model)
#######################################################################
#                 PART 2  Inference/Evaluation Configuration          #
#######################################################################

judge_abbr = 'CompassJudger-1-32B-Instruct'
judge_ip = 'dlcaab18nscx0n16-master-0:23333'
judge_url = f'http://{judge_ip}/v1/chat/completions'
judge_path = 'opencompass/CompassJudger-1-32B-Instruct'

aliyun_cfg = dict(
    python_env_path='/cpfs01/user/chenyicheng/miniconda3/envs/jize',
    dlc_config_path='/cpfs01/user/chenyicheng/.dlc/config',
    workspace_id='ws1ujs9hjg4li05e',
    worker_image='pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/pjlab-eflops/lingjun-pytorch-training:2.3-24.03-gu8-gpu-py310-cu124-ubuntu22.04',
    resource_id='',
    data_sources=[],
    hf_offline=False,
    bashrc_path='/cpfs01/user/chenyicheng/wangjize/.bashrc',
    conda_env_name='jize',
    # optional, suggest to set the http_proxy if `hf_offline` if False.
    # http_proxy="http://closeai-proxy.pjlab.org.cn:23128",
    # optional, using mirror to speed up the huggingface download
    hf_endpoint='https://hf-mirror.com',
    # optional, if not set, will use the default cache path
    huggingface_cache=
    '/cpfs01/shared/public/opencompass/models/hf_hub',
    # torch_cache='/cpfs01/shared/public/public_hdd/llmeval/model_weights/torch',
    extra_envs=[
        # 'LD_LIBRARY_PATH=/cpfs01/shared/public/zhaoqian/cuda-compat-12-2:$LD_LIBRARY_PATH',
        # 'COMPASS_DATA_CACHE=/cpfs01/user/wangjize/.cache/opencompass',
        'COMPASS_DATA_CACHE=/cpfs01/shared/public/llmeval/compass_data_cache',
        f'NO_PROXY="{judge_ip},hf-mirror.com,openaipublic.blob.core.windows.net,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1"',
        f'no_proxy="{judge_ip},hf-mirror.com,openaipublic.blob.core.windows.net,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1"',
        # 'TIKTOKEN_CACHE_DIR=/cpfs01/shared/public/public_hdd/llmeval/share_tiktoken',
        # "VLLM_USE_MODELSCOPE=False",
    ],
    dlc_job_cmd="create",
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=DLCRunner,
        max_num_workers=64,
        retry=0,  # Modify if needed
        aliyun_cfg=aliyun_cfg,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=aliyun_cfg,
        max_num_workers=128,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLEvalTask),
    ),
)

# _mmlu_biomed = ['anatomy', 'college_biology', 'clinical_knowledge', 'college_medicine', 'medical_genetics', 'professional_medicine']
# _mmlu_biomed = ['lukaemon_mmlu_' + s for s in _mmlu_biomed]

summarizer = dict(
    dataset_abbrs=[  
        'math',
        'gsm8k',
        'ARC-c',
        'sanitized_mbpp',
        'race-high',
        # 'mmlu', 
        'mmlu-biomed',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

# summarizer = dict(
#     dataset_abbrs=[  
#         'math-test',
#         'gsm8k-test',
#         'arc-c-test',
#         'mbpp-test',
#         'race-high-test',
#         'mmlu',    
#         'mmlu-biomed',
#     ],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []
#     ),
# )


work_dir = '/cpfs01/user/chenyicheng/wangjize/opencompass/outputs/smoa'