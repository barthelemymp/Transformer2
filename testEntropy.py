# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:24:37 2021

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
torch.set_num_threads(16)
print("import done")
#torch.functional.one_hot
pathtoFolder = "/home/feinauer/Datasets/DomainsInter/processed/"
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
alphalist=[-0.1,0.0, 0.1]
wd_list = [0.0]#, 0.00005]
# ilist = [46, 69, 71,157,160,251, 258, 17]
onehot=False
wd=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
import sys
family = str(sys.argv[1])
i = int(family)

for alpha in alphalist:

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
    ntrain = len(pds_train)
    nval = len(pds_val)
    ntest = len(pds_test)
    dval1,dval2 = distanceTrainVal(pds_train, pds_val)
    print("median", (dval1+dval2).min(dim=0)[0].median())
    maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
    maskValclose = maskValclose.cpu().numpy()
    maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
    maskValfar = maskValfar.cpu().numpy()
    # ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
    # print("score", i)
    # print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

    train_iterator = DataLoader(pds_train, batch_size=batch_size,
                    shuffle=True, num_workers=0, collate_fn=default_collate)
    test_iterator = DataLoader(pds_test, batch_size=batch_size,
                    shuffle=True, num_workers=0, collate_fn=default_collate)
    val_iterator = DataLoader(pds_val, batch_size=batch_size,
                    shuffle=True, num_workers=0, collate_fn=default_collate)
    
    
    # Model hyperparameters
    
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
    
    #whyyy 'cpu?'
    wandb.init(project="Transformer test Entropy", entity="barthelemymp")
    config_dict = {
      "num_layers": num_encoder_layers,
      "embedding":embedding_size,
      "forward_expansion": forward_expansion,
      "batch_size": batch_size,
      "Encoder": "Positional",
      "Family":i,
      "dropout":dropout,
      "len input":len_input,
      "len output":len_output,
      "sizetrain": len(pds_train),
      "sizeval": len(pds_val),
      "num_heads": num_heads,
      "loss": "CE",
      "alpha":alpha,
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
            
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    sparseoptim = torch.optim.SGD(model.parameters(), lr=5e-6)
    pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
    for epoch in range(num_epochs+1):
        print(f"[Epoch {epoch} / {num_epochs}]")
        model.train()
        lossesCE = []
        accuracyTrain = 0
        for batch_idx, batch in enumerate(train_iterator):
            if epoch<3000:
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
            else:
                optimizer.zero_grad()
                lossCE, lossEntropy, acc = ConditioalEntropyMatchingLoss(batch, model, criterion, device, samplingMultiple=10)
                accuracyTrain += acc
                lossesCE.append(lossCE.item())
                loss = lossCE + alpha * lossEntropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                
                sparseoptim.step()
            
            
        mean_lossCETrain = sum(lossesCE) / len(lossesCE)
        accuracyTrain = accuracyTrain/ntrain
        # mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
        step += 1
        # scheduler.step(mean_lossCETrain)
        model.eval()
        lossesCE_eval = []
        lossesMatching_eval = []
        accuracyVal = 0
        
        if epoch%1==0:
            with  torch.no_grad():
                for batch_idx, batch in enumerate(val_iterator):
                    inp_data, target= batch[0], batch[1]
                    inp_data = inp_data.to(device)
                    output = model(inp_data, target[:-1, :])
                    accuracyVal += accuracy(batch, output, onehot=False).item()
                    output = output.reshape(-1, output.shape[2]) #keep last dimension
                    if onehot:
                        _, targets_Original = target.max(dim=2)
                    else:
                        targets_Original= target
                    targets_Original = targets_Original[1:].reshape(-1)
                    loss_eval = criterion(output, targets_Original)
                    lossesCE_eval.append(loss_eval.item()) 
                mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
                accuracyVal = accuracyVal/nval
                step_ev +=1
        #wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal,  "accuracyVal":accuracyVal ,  "accuracyTrain": accuracyTrain, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
        
        
        lossesCE_test = []
        lossesMatching_test = []
        accuracytest = 0
        
        if epoch%1==0:
            with  torch.no_grad():
                for batch_idx, batch in enumerate(test_iterator):
                    inp_data, target= batch[0], batch[1]
                    inp_data = inp_data.to(device)
                    output = model(inp_data, target[:-1, :])
                    accuracytest += accuracy(batch, output, onehot=False).item()
                    output = output.reshape(-1, output.shape[2]) #keep last dimension
                    if onehot:
                        _, targets_Original = target.max(dim=2)
                    else:
                        targets_Original= target
                    targets_Original = targets_Original[1:].reshape(-1)
                    loss_test = criterion(output, targets_Original)
                    lossesCE_test.append(loss_test.item()) 
                mean_losstest = sum(lossesCE_test) / len(lossesCE_test)
                accuracytest = accuracytest/ntest
                step_ev +=1
        wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "test loss CE": mean_losstest,  "accuracyVal":accuracyVal , "accuracytest":accuracytest ,  "accuracyTrain": accuracyTrain, "epoch":epoch})
        
        
        if epoch%200==0:
            criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
            model.eval()
            criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
            scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
            scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
            scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
            scoreMatchingValClose = sum((scoHVal[0]==scoHVal[1])[maskValclose])
            scoreMatchingValFar = sum((scoHVal[0]==scoHVal[1])[maskValfar])
            # scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
            # scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
            # scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
            Entropy = ConditionalEntropyEstimatorGivenInp(pds_val[0][0], model, pds_train.SymbolMap["<pad>"], len_output,nseq=100000, batchs=1000, returnAcc=False)
            
            wandb.log({ "scoreMatching Val": scoreMatchingVal, "scoreMatchingValClose": scoreMatchingValClose, "scoreMatchingVal Far": scoreMatchingValFar, "Entropy":Entropy, "epoch":epoch})
    wandb.finish()
        
        
        

# i = "hk"
# pathTofile = 'train_real.csv'
# inputsize, outputsize = getLengthfromCSV(pathTofile)
# os.path.isfile(pathTofile)
# count +=1
# print("ddi", i, " is running")
# name = "combined_MSA_ddi_" +str(i)+"_joined"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #Dataset
# train_path = 'train_real.csv'
# val_path = 'val_real.csv'
# test_path = 'test_real.csv'
# #add 2 for start and end token 
# len_input = inputsize + 2
# len_output =outputsize + 2

# pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
# print("score", )
# print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

    
    






