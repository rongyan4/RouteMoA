from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2, normalize_final_answer

math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    # dict(
    #     type=MATHDataset,
    #     abbr='math-train',
    #     path='/cpfs01/user/chenyicheng/wangjize/emoa/data/math/repeat/',
    #     file_name='math_train_rp5.json',
    #     reader_cfg=math_reader_cfg,
    #     infer_cfg=math_infer_cfg,
    #     eval_cfg=math_eval_cfg,
    # ),
    # dict(
    #     type=MATHDataset,
    #     abbr='math-dev',
    #     path='/cpfs01/user/chenyicheng/wangjize/emoa/data/math/repeat/',
    #     file_name='math_dev_rp5.json',
    #     reader_cfg=math_reader_cfg,
    #     infer_cfg=math_infer_cfg,
    #     eval_cfg=math_eval_cfg,
    # ),
    dict(
        type=MATHDataset,
        abbr='math-test',
        path='/cpfs01/user/chenyicheng/wangjize/emoa/data/math/repeat/',
        file_name='math_test_rp5.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    ),
]