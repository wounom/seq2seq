import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers,batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers,batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        #prediction = self.fc_out(output)
        return prediction, hidden

# 定义seq2seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[1]
        batch_size = trg.shape[0]
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size,trg_len, output_dim).to(self.device)

        hidden = self.encoder(src)

        # 解码器的输入为目标序列的第一个词
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            
            output, hidden = self.decoder(input, hidden)
            outputs[:,t] = output
            
            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:,t] if teacher_force else output.argmax(1)

        return outputs 