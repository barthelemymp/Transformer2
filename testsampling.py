# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:49:56 2022

@author: bartm
"""
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
import wandb
print("wandb imported")
wandb.login()
print("wandb login")
sys.path.append("")
from ProteinTransformer import *
from ProteinsDataset import *
from MatchingLoss import *
from utils import *
from ardca import *
print("import done")
#torch.functional.one_hot
pathtoFolder = "/home/Datasets/DomainsInter/processed/"
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 5
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 25#len(protein.vocab) 
trg_vocab_size = 25#len(protein_trans.vocab) 
embedding_size = 55#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token

repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.01, 0.1]
wd_list = [0.0]#, 0.00005]
# ilist = [46, 69, 71,157,160,251, 258, 17]
onehot=False
wd=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


i=46
pathTofile = "testsampleRange.csv"
inputsize, outputsize = getLengthfromCSV(pathTofile)
os.path.isfile(pathTofile)
count +=1
print("ddi", i, " is running")
name = "combined_MSA_ddi_" +str(i)+"_joined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Dataset

len_input = inputsize + 2
len_output =outputsize + 2
pds = ProteinTranslationDataset(pathTofile, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
train_iterator = DataLoader(pds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=default_collate)
ntrain = len(pds)


src_pad_idx = pds.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
src_position_embedding = PositionalEncoding(embedding_size, max_len=len_input,device=device)
trg_position_embedding = PositionalEncoding(embedding_size, max_len=len_output, device=device)
        
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    src_position_embedding,
    trg_position_embedding,
    device,
    onehot=onehot,
).to(device)


step = 0
step_ev = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 5e-4

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)
        
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds.SymbolMap["<pad>"])
for epoch in range(100):
    print(f"[Epoch {epoch} / {num_epochs}]")
    _ = model.train()
    lossesCE = []
    accuracyTrain = 0
    for batch_idx, batch in enumerate(train_iterator):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        accuracyTrain += accuracy(batch, output, onehot=False).item()
        output = output.reshape(-1, output.shape[2])#keep last dimension
        if onehot:
            _, targets_Original = target.max(dim=2)
        else:
            targets_Original= target
        targets_Original = targets_Original[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, targets_Original)
        lossesCE.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()
    mean_lossCETrain = sum(lossesCE) / len(lossesCE)
    accuracyTrain = accuracyTrain/ntrain
    Entropy = ConditionalEntropyEstimator(pds, model, batchs=100)
    print(mean_lossCETrain, accuracyTrain, Entropy)