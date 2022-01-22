# -*- coding: utf-8 -*-

"""
Created on Tue Jan 11 18:10:31 2022

@author: bartm
"""
##ffffwffff
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

count = 0 
repartition = [0.7, 0.15, 0.15]
ilist = []

for i in range(300):
    pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
    if os.path.isfile(pathTofile)==True:
        ilist.append(i)
        print(i)
        name = "combined_MSA_ddi_" +str(i)+"_joined"
        train_path = pathtoFolder + name +'_train.csv'


# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 5
dropout = 0.10
forward_expansion = 2048
      
batch_size = 32
num_heads = 5
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.10
src_vocab_size = 25
trg_vocab_size = 25
embedding_size = 555



#EPOCHS 
num_epochs =5000
Unalign = False


onehot =False


ifirst = ilist[0]
pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(ifirst)+"_joined.csv"
inputsize, outputsize = getLengthfromCSV(pathTofile)
os.path.isfile(pathTofile)
count +=1
print("ddi", ifirst, " is running")
name = "combined_MSA_ddi_" +str(ifirst)+"_joined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = pathtoFolder + name +'_train.csv'
val_path = pathtoFolder + name +'_val.csv'
test_path = pathtoFolder + name +'_test.csv'
len_input = inputsize + 2
len_output =outputsize + 2

pds_train_T = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
pds_test_T = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
pds_val_T = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)



name17 = "combined_MSA_ddi_" +str(17)+"_joined"
val_path17 = pathtoFolder + name17 +'_val.csv'
pds_val17 = ProteinTranslationDataset(val_path17, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
nval17 = len(pds_val17)
val_iterator17 = DataLoader(pds_val17, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)  


# name46 = "combined_MSA_ddi_" +str(46)+"_joined"
# val_path46 = pathtoFolder + name46 +'_val.csv'
# pds_val46 = ProteinTranslationDataset(val_path17, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# nval46 = len(pds_val46)
# val_iterator17 = DataLoader(pds_val_T, batch_size=batch_size,
#                 shuffle=True, num_workers=0, collate_fn=default_collate)  





src_pad_idx = pds_val17.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
lenilist = []
lenolist = []
trainlist = []
for i in ilist[1:]:
    pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
    inputsize, outputsize = getLengthfromCSV(pathTofile)
    os.path.isfile(pathTofile)
    count +=1
    print("ddi", i, " is running")
    name = "combined_MSA_ddi_" +str(i)+"_joined"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = pathtoFolder + name +'_train.csv'
    val_path = pathtoFolder + name +'_val.csv'
    test_path = pathtoFolder + name +'_test.csv'
    len_input = inputsize + 2
    len_output =outputsize + 2
    lenilist.append(len_input)
    lenolist.append(len_output)
    if len_input<300:
        if len_output<300:
            try:
                pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
                # pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
                pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
                pds_train_T.join(pds_train)
                pds_val_T.join(pds_val)
            except:
                print("family", i, " is to be cleaned")
                

len_input = pds_train_T[0][0].shape[0]
len_output = pds_train_T[0][1].shape[0]
src_position_embedding = PositionalEncoding(embedding_size, max_len=len_input,device=device)
trg_position_embedding = PositionalEncoding(embedding_size, max_len=len_output, device=device)
     

train_iterator = DataLoader(pds_train_T, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)

val_iterator = DataLoader(pds_val_T, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)    

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

wandb.init(project="Full Domains", entity="barthelemymp")
config_dict = {
  "num_layers": num_encoder_layers,
  "embedding":embedding_size,
  "forward_expansion": forward_expansion,
  "batch_size": batch_size,
  "Encoder": "Positional",
  "Family":"All",
  "dropout":dropout,
  "len input":len_input,
  "len output":len_output,
  "scheduler": "none",
  "loss": "CE",
  "weight_decay":0.0
}
wandb.config.update(config_dict) 



  
step = 0
step_ev = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 5e-5

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)


scaler = torch.cuda.amp.GradScaler()
  
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
for epoch in range(num_epochs+1):
    print(f"[Epoch {epoch} / {num_epochs}]")
    model.train()
    lossesCE = []
    accuracyTrain = 0
    for batch_idx, batch in enumerate(train_iterator):
#        print(batch_idx)
#        with torch.cuda.amp.autocast():
        optimizer.zero_grad()
        inp_data, target= batch[0].long(), batch[1].long()
        output = model(inp_data, target[:-1, :])
        # accuracyTrain += accuracy(batch, output, onehot=False).item()
        output = output.reshape(-1, output.shape[2])#keep last dimension
        if onehot:
            _, targets_Original = target.max(dim=2)
        else:
            targets_Original = target
        targets_Original = targets_Original[1:].reshape(-1)
        loss = criterion(output, targets_Original)
#        print(loss)
        lossesCE.append(loss)
        # scaler.scale(loss).backward()
        loss.backward()
        # scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        # scaler.step(optimizer)
        optimizer.step()
        # scaler.update()
    mean_lossCETrain = sum(lossesCE).item() / len(lossesCE)

    # mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
    step += 1
    # scheduler.step(mean_lossCETrain)
    model.eval()
    lossesCE_eval = []
    lossesMatching_eval = []
    #accuracyVal = 0
    if epoch%1==0:
        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
#                print(batch_idx)
                inp_data, target = batch[0].long(), batch[1].long()
                inp_data = inp_data.to(device)
                output = model(inp_data, target[:-1, :])
                #accuracyVal += accuracy(batch, output, onehot=False).item()
                output = output.reshape(-1, output.shape[2]) #keep last dimension
                if onehot:
                    _, targets_Original = target.max(dim=2)
                else:
                    targets_Original = target
                targets_Original = targets_Original[1:].reshape(-1)
                loss_eval = criterion(output, targets_Original)
                lossesCE_eval.append(loss_eval) 
            mean_lossVal = sum(lossesCE_eval).item() / len(lossesCE_eval)
            #accuracyVal = accuracyVal/nval
            step_ev += 1
            
            
            
            
        lossesCE_eval17 = []
        lossesMatching_eval17 = []
        accuracyVal17 = 0

        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator17):
                inp_data, target = batch[0].long(), batch[1].long()
                inp_data = inp_data.to(device)
                output = model(inp_data, target[:-1, :])
                accuracyVal17 += accuracy(batch, output, onehot=False).item()
                output = output.reshape(-1, output.shape[2]) #keep last dimension
                if onehot:
                    _, targets_Original = target.max(dim=2)
                else:
                    targets_Original = target
                targets_Original = targets_Original[1:].reshape(-1)
                loss_eval = criterion(output, targets_Original)
                lossesCE_eval17.append(loss_eval.item()) 
            mean_lossVal17 = sum(lossesCE_eval17) / len(lossesCE_eval17)
            accuracyVal17 = accuracyVal17/nval17
            step_ev += 1
            wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "Val loss CE 17": mean_lossVal17, "accuracyVal17":accuracyVal17, "epoch":epoch})
            
    if epoch%50==49:
        model.eval()
        criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        scoreHungarianVal = HungarianMatchingBS(pds_val17, model,100)
        scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
        scoreMatchingVal17 = sum(scoHVal[0]==scoHVal[1])

        # scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
        # scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
        # scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
        wandb.log({ "scoreMatching Val17": scoreMatchingVal17, "epoch":epoch})
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="CompleteDomainsTransformer.pth.tar")

    