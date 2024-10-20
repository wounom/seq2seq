import random
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
from data import prepare_data
from model import Decoder, Encoder, Seq2Seq
from torch.nn.utils.rnn import pad_sequence
import time

SOS_token = 0
EOS_token = 1
PAD_token = 2  # 用于填充
USE_CUDA = torch.cuda.is_available()

# 准备数据
input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)
input_vocab_size = len(input_lang.index2word)
output_vocab_size = len(output_lang.index2word)

print(f"Input vocabulary size: {input_vocab_size}")
print(f"Output vocabulary size: {output_vocab_size}")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建训练数据
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)  # 添加 EOS_token
    var = torch.LongTensor(indexes)
    if USE_CUDA:
        var = var.cuda()  # 转移到 GPU
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

train_data = [variables_from_pair(pair) for pair in pairs]
input_variables, target_variables = zip(*train_data)

# 使用 pad_sequence 填充输入和目标张量
input_tensor = pad_sequence(input_variables, batch_first=True, padding_value=PAD_token )  # [batch_size ,max_length]
target_tensor = pad_sequence(target_variables, batch_first=True, padding_value=PAD_token )  #[batch_size ,max_length]

# 创建 DataLoader
dataset = TensorDataset(input_tensor, target_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = Encoder(len(input_lang.index2word), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
dec = Decoder(len(output_lang.index2word), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
model = Seq2Seq(enc, dec, device).to(device)

# 训练设置
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# 训练函数
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for src, trg in iterator:
        src, trg = src.to(device), trg.to(device)
       
        optimizer.zero_grad()

        intrg=trg[:,:-1]
        # 创建一个全0的Tensor，其形状与第二维的每个子Tensor相同，除了第一个维度是1
        zero_tensor = torch.zeros((intrg.size(0), 1)).long().to(device)
        
        # 使用torch.cat在第二维（dim=1）上拼接这两个Tensor
        resultin_trg = torch.cat((zero_tensor, intrg), dim=1)
        
        output = model(src, resultin_trg)#[batch_size,sequence_len,output_dim]
        logp = nn.functional.log_softmax(output, dim=-1)
        

        output_dim = output.shape[-1]
        
        logp = logp.reshape(-1, output_dim)  # 使用 reshape
        trg = trg.reshape(-1)  # 使用 reshape

        loss = criterion(logp, trg)  # 计算损失
        #print(loss)
        loss.backward()
       # print(output.size())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# 开始训练
N_EPOCHS = 10
CLIP = 1

for epoch in range(1, N_EPOCHS + 1):
    start = time.time()
    loss = train(model, train_loader, optimizer, criterion, CLIP)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    end = time.time()
    print(end-start)
