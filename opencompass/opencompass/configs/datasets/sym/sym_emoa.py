from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess, match_answer_pattern
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sym import SYMDataset_for_emoa, SYMDataset_for_emoa_mbpp
from opencompass.datasets import MATHEvaluator, math_postprocess_v2, MATHEvaluator_emoa
from opencompass.datasets import EmoaGSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator, gsm8k_dataset_postprocess_for_sym_emoa
from opencompass.datasets.mbpp import EmoaSanitizedMBPPDataset, MBPPEvaluator


sym_reader_cfg = dict(
    input_columns=['prompt','test_list'],
    output_column='test_list_2'
)
sym_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN',prompt='{prompt}'),
                dict(role='BOT',prompt='')]
                )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
sym_eval_cfg = dict(
    evaluator=dict(type=MBPPEvaluator), 
    pred_role='BOT'
    )
"""
dataset     infer   eval
arc-c       ok      ok
gsm8k       ok      ok
math        ok      ok
mbpp        ok      
mmlu        ok      ok
race_high   ok      ok

"""
dataset_name = "mbpp_train"
sym_datasets = [
    dict(
        abbr=dataset_name,
        type=SYMDataset_for_emoa_mbpp,
        name=dataset_name,
        path=f'/cpfs01/user/chenyicheng/wangjize/songyiming/dataset/{dataset_name}.jsonl',
        reader_cfg = sym_reader_cfg,
        infer_cfg = sym_infer_cfg,
        eval_cfg = sym_eval_cfg,

    )
]