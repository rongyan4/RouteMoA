import json

class OracleRouter:
    def __init__(self, file_path='/cpfs01/user/chenyicheng/wangjize/emoa/src/emoa/router/data/processed/routerset_test/test.json'):
        self.type = 'oracle'
        self.database = []
        with open(file_path, 'r', encoding='utf-8') as f:
            self.database = json.load(f)
        self.models = [
            "google/gemma-2-9b-it",
            "mistralai/Ministral-8B-Instruct-2410",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Math-7B-Instruct",
            "ContactDoctor/Bio-Medical-Llama-3-8B"
        ]
    def forward(self, query):
        for item in self.database:
            if item['query'].strip() in query:
                preds = dict()
                for model in self.models:
                    preds[model] = item['labels'][model]
                return preds
        return None
