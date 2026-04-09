import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class MLPRouter(nn.Module):
    def __init__(self, pretrained_bert, hidden_dim, num_labels):
        """
        Initialize MLP Router model

        :param pretrained_bert: pre-trained BERT model
        :param hidden_dim: hidden layer dimension
        :param num_labels: number of output labels
        """
        super(MLPRouter, self).__init__()

        # Define BERT model
        self.bert = AutoModel.from_pretrained(pretrained_bert)
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Get BERT hidden dimension
        self.input_dim = self.bert.config.hidden_size

        # Define Multi-Layer Perceptron (MLP)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden to hidden layer
        self.fc3 = nn.Linear(hidden_dim, num_labels)  # Hidden to output layer
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass: through MLP network

        :param input_ids: input token IDs
        :param attention_mask: attention mask
        """
        # Get BERT output
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation as sentence-level embedding
        x = outputs.pooler_output  # shape: (batch_size, hidden_size)

        # MLP part
        x = F.relu(self.fc1(x))  # First layer activation function
        x = self.dropout(x)  # Dropout
        x = F.relu(self.fc2(x))  # Second layer activation function
        x = self.dropout(x)  # Dropout
        x = self.fc3(x)  # Output layer

        return x
