import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class RouterModel(nn.Module):
    def __init__(self, pretrained_bert, num_labels):
        super(RouterModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_bert)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden_state)
        return logits
