import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class MatrixFactorizationRouter(nn.Module):
    def __init__(self, pretrained_bert, num_labels):
        super(MatrixFactorizationRouter, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert)
        self.model_embeddings = torch.nn.Embedding(num_labels, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    
    def compute_similarity(self, input1, input2):  # input1: (batch_size, hidden_size), input2: (num_models, hidden_size)
        return (input1 @ input2.T) # / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # (batch_size, hidden_size)
        prompt_embedding = outputs.pooler_output  # (num_models, hidden_size)
        logits = self.compute_similarity(prompt_embedding, self.model_embeddings.weight)  # (batch_size, num_models)
        # prompt_embedding = F.normalize(prompt_embedding, p=2, dim=1).unsqueeze(1) # (batch_size, 1, hidden_size)
        # model_embedding = F.normalize(self.model_embeddings.weight, p=2, dim=1).unsqueeze(0) # (1, num_models, hidden_size)
        # logits = self.classifier(prompt_embedding * model_embedding).squeeze(2) # (batch_size, num_models)
        # breakpoint()
        return logits
