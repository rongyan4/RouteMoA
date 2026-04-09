from opencompass.datasets.base import BaseDataset
# from .base import BaseDataset
import datasets
import jsonlines
import random

agg_prompt = """
You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
"""
# The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

class SYMDataset_for_emoa(BaseDataset):
        @staticmethod
        def load(name,path:str, prompt:str=agg_prompt, origin_question:bool=True, **kwargs) ->datasets.Dataset:
            data = []
            with jsonlines.open(path, 'r') as reader:
                for row in reader: # 对于一个问题
                    # 拼接prompt
                    if origin_question == True:
                        extended_prompt = prompt+'\n'+'\n'+'This is the original question answered by these models:'+'\n'+f"{row['prompt']}"+'\n'+'\n'+'Responses from models:'+'\n'
                    elif origin_question == False:
                        extended_prompt = prompt+'\n'+'\n'+'Responses from models:'+'\n'
                    
                    for i, model_response in enumerate(row['origin_prediction'].values()):
                        if model_response != ' ':
                            extended_prompt += f'{i+1}.{model_response}'+'\n' 
                    extended_prompt = extended_prompt[:-1]
                    #把拼接完的prompt转换为元数据
                    metadata = {
                        'source':row['source'],
                        'id':row['id'],
                        'answerKey':row['answerKey'],
                        'prompt':extended_prompt
                    }
                    data.append(metadata)
            return datasets.Dataset.from_list(data)

class SYMDataset_for_emoa_mbpp(BaseDataset):
        @staticmethod
        def load(name,path:str, prompt:str=agg_prompt, origin_question:bool=True,test_list:int=2, return_list:bool=False,**kwargs) ->datasets.Dataset:
            data = []
            with jsonlines.open(path, 'r') as reader:
                for row in reader: # 对于一个问题
                    # 拼接prompt
                    if origin_question == True:
                        extended_prompt = prompt+'\n'+'\n'+'This is the original question answered by these models:'+'\n'+f"{row['prompt']}"+'\n'+'\n'+'Responses from models:'+'\n'
                    elif origin_question == False:
                        extended_prompt = prompt+'\n'+'\n'+'Responses from models:'+'\n'
                    
                    for i, model_response in enumerate(row['origin_prediction'].values()):
                        if model_response != ' ':
                            extended_prompt += f'{i+1}.{model_response}'+'\n' 
                    extended_prompt = extended_prompt[:-1]
                    #把拼接完的prompt转换为元数据
                    metadata = {
                        'source':row['source'],
                        'id':row['id'],
                        'test_list':row['answerKey'],
                        'test_list_2':row['answerKey'][test_list],
                        'prompt':extended_prompt
                    }
                    data.append(metadata)
            if not return_list:
                return datasets.Dataset.from_list(data)
            else:
                return data