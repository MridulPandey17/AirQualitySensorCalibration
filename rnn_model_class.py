import os 
import sys 
import torch
import numpy as np
import pickle
import torch.nn as nn 
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, use_pretrained_embeddings:bool = None, no_layers = None, vocab_size = None,
        hidden_dim = None, embedding_dim = None, drop_prob=0.3, output_dim = None, n_classes = None):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.use_pretrained_embeddings = use_pretrained_embeddings
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if not use_pretrained_embeddings else None
        
        #lstm
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = self.hidden_dim,
                            num_layers = no_layers, batch_first = True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
    
        # linear  layer
        self.fc1 = nn.Linear(self.hidden_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, n_classes)
        
    def forward(self, x, hidden = None):
        
        # x.shape = (batch_size, seq_len, embedding_dim) : If use_pretrained_embeddings = True
        # x.shape = (batch_size, seq_len) : If use_pretrained_embeddings = False
        
        batch_size = x.size(0)
        if hidden is None: 
            hidden = self.init_hidden(batch_size)
        
        if x.device != hidden[0].device:
            hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
        
        if not self.use_pretrained_embeddings:
            x = self.embedding(x)
        
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out.shape = (batch_size, seq_len, hidden_dim)
        # print(f"lstm_out.shape = {lstm_out.shape}, hidden[0].shape = {hidden[0].shape}, hidden[1].shape = {hidden[1].shape}")
        # print(f"lstm_out.device = {lstm_out.device}")
        
        lstm_out = lstm_out.contiguous()[:, -1, :]
        # lsmt_out.shape = (batch_size, hidden_dim)
        
        # out = self.dropout(F.tanh(lstm_out))
        # out = F.tanh(lstm_out)
        out = self.dropout(lstm_out)
        out = F.relu(self.fc1(lstm_out))
        # out = self.fc1(out)
        out = self.fc2(out)
        return out
        
        
    def init_hidden(self, batch_size):
        
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.randn((self.no_layers, batch_size, self.hidden_dim))
        c0 = torch.randn((self.no_layers, batch_size, self.hidden_dim))
        hidden = (h0, c0)
        return hidden

# if __name__ == "__main__":
    
#     device = torch.device("mps:0")
    
#     model = RNN(use_pretrained_embeddings=True, no_layers=1, vocab_size=10000, hidden_dim=100, embedding_dim=100, output_dim=100,
#         n_classes=2)
    
#     model.to(device)
    
#     for name, param in model.named_parameters():
#         print(f"name = {name}, device = {param.device}, shape = {param.shape}")
    
    
#     with open("data/data_pretrained_embeddings/np_train_input_80.pkl", "rb") as f:
#         np_train_input = pickle.load(f)
        
#     tensor_train_input = torch.from_numpy(np_train_input).to(device)
#     print(f"tensor_train_input.shape = {tensor_train_input.shape}")
#     print(f"tensor_train_input.dtype = {tensor_train_input.dtype}")
#     print(f"tensor_train_input.device = {tensor_train_input.device}")
    
#     output = model(tensor_train_input)
#     print(f"output.shape = {output.shape}")