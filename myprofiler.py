
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler


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

sys.path.append("")
from ProteinTransformer import *
from ProteinsDataset import *
from MatchingLoss import *
from utils import *
from ardca import *


import sys
family = str(sys.argv[1])
i = int(family)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onehot=False
num_epochs = 500
Unalign = False
count=0
i = 17
##### Training simple 

pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
inputsize, outputsize = getLengthfromCSV(pathTofile)
os.path.isfile(pathTofile)
count +=1
print("ddi", i, " is running")
name = "combined_MSA_ddi_" +str(i)+"_joined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Dataset
train_path = pathtoFolder + name +'_train.csv'
val_path = pathtoFolder + name +'_val.csv'
test_path = pathtoFolder + name +'_test.csv'

#add 2 for start and end token 
len_input = inputsize + 2
len_output =outputsize + 2

pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
print("score", i)
print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

train_iterator = DataLoader(pds_train, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
test_iterator = DataLoader(pds_test, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
val_iterator = DataLoader(pds_val, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)


# Model hyperparameters
src_vocab_size = 25#len(protein.vocab) 
trg_vocab_size = 25#len(protein_trans.vocab) 
embedding_size = 255#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
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
learning_rate = 3e-4

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
for epoch in range(num_epochs+1):
    print(f"[Epoch {epoch} / {num_epochs}]")
    model.train()
    lossesCE = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
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
    # mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
    step += 1
    # scheduler.step(mean_lossCETrain)
    model.eval()
    lossesCE_eval = []
    lossesMatching_eval = []
    if epoch%1==0:
        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                inp_data, target= batch[0], batch[1]
                inp_data = inp_data.to(device)
                output = model(inp_data, target[:-1, :])
                output = output.reshape(-1, output.shape[2]) #keep last dimension
                if onehot:
                    _, targets_Original = target.max(dim=2)
                else:
                    targets_Original= target
                targets_Original = targets_Original[1:].reshape(-1)
                loss_eval = criterion(output, targets_Original)
                lossesCE_eval.append(loss_eval.item()) 
            mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
            step_ev +=1
    wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "alpha":alpha, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
    if epoch%100==0:
        criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        model.eval()
        criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
        scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
        scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
        scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
        scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
        scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
        wandb.log({"scoreMatching Train": scoreMatchingTrain, "scoreMatching Val": scoreMatchingVal, "epoch":epoch})
wandb.finish()


