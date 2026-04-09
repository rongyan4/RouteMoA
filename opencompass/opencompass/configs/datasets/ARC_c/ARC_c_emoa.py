from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess, match_answer_pattern

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A. {textA}
B. {textB}
C. {textC}
D. {textD}
""".strip()

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey')

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=QUERY_TEMPLATE)
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

ARC_c_datasets = [
    # dict(
    #     abbr='arc-c-train',
    #     type=ARCDataset,
    #     path='/cpfs01/user/chenyicheng/wangjize/emoa/data/arc-c/repeat/arc-c_train_rp5.jsonl',
    #     name='ARC-Challenge',
    #     reader_cfg=ARC_c_reader_cfg,
    #     infer_cfg=ARC_c_infer_cfg,
    #     eval_cfg=ARC_c_eval_cfg,
    # ),
    # dict(
    #     abbr='arc-c-dev',
    #     type=ARCDataset,
    #     path='/cpfs01/user/chenyicheng/wangjize/emoa/data/arc-c/repeat/arc-c_dev_rp5.jsonl',
    #     name='ARC-Challenge',
    #     reader_cfg=ARC_c_reader_cfg,
    #     infer_cfg=ARC_c_infer_cfg,
    #     eval_cfg=ARC_c_eval_cfg,
    # ),
    dict(
        abbr='arc-c-test',
        type=ARCDataset,
        path='/cpfs01/user/chenyicheng/wangjize/emoa/data/arc-c/repeat/arc-c_test_rp5.jsonl',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
