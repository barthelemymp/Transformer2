# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:38:18 2021

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

sys.path.append("")
from ProteinTransformer import *
from ProteinsDataset import *
from MatchingLoss import *
from utils import *
from ardca import *


import sys
family = str(sys.argv[1])
i = int(family)

import wandb
wandb.login()

sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'Val loss CE',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric


# early_terminate={
#   "type": "hyperband",
#   "min_iter": 100
#   } 

# sweep_config['early_terminate'] = early_terminate
datasettype= "Domains"


parameters_dict = {
    # 'optimizer': {
    #     'values': ['adam', 'sgd']
    #     },
    'forward_expansion': {
        'values': [2048]
        },
    'optim': {
          'values': ["adam"]
        },
    'lr': {
          'values': [0.001, 0.0001, 0.00001]
        },
    
    'precision': {
          'values': ["classic"]
        },
    
    'num_heads': {
          'values': [5]
        },
    'batch_size': {
          'values': [32]
        },
    'num_layer': {
          'values': [2]
        },
    
    'embedding_size': {
          'values': [55]
        },
    
    'dropout': {
          'values': [0.1]
        },
    
    'weight_decay': {
          'values': [0.0]
        },
    
    'fam': {
          'values': [datasettype+str(i)]
        },
    }

sweep_config['parameters'] = parameters_dict
repartition = [0.7, 0.15, 0.15]

#torch.functional.one_hot
def train(config=None):
    scaler = torch.cuda.amp.GradScaler()
      
    with wandb.init(config=config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = wandb.config
        onehot=False
        num_epochs = 3000
        Unalign = False
        count=0
        if datasettype == "Domains":
            pathtoFolder = "/home/Datasets/DomainsInter/processed/"
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
        if datasettype == "PPI":
            pathtoFolder = "/home/Datasets/DomainsInter/PPIprocessed/"
            pathTofile = pathtoFolder+ "PPI_" +str(i)+"_joined.csv"
            inputsize, outputsize = getLengthfromCSV(pathTofile)
            os.path.isfile(pathTofile)
            count +=1
            print("PPI", i, " is running")
            name = "PPI_" +str(i)+"_joined"
            #splitcsv(pathtoFolder, name, repartition, shuffle=True, maxval=500)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #Dataset
            train_path = pathtoFolder + name +'_train.csv'
            val_path = pathtoFolder + name +'_val.csv'
            test_path = pathtoFolder + name +'_test.csv'
        if datasettype == "HKRR":
            pathtoFolder = ""
            pathTofile = 'train_real.csv'
            inputsize, outputsize = getLengthfromCSV(pathTofile)
            os.path.isfile(pathTofile)
            count +=1
            print("HKRR", " is running")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #Dataset
            train_path = 'train_real.csv'
            val_path = 'val_real.csv'
            test_path = 'test_real.csv'
        count = 0

        # Model hyperparameters--> CAN BE CHANGED

        #add 2 for start and end token 
        len_input = inputsize + 2
        len_output =outputsize + 2
        
        pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        ntrain = len(pds_train)
        nval = len(pds_val)
        dval1,dval2 = distanceTrainVal(pds_train, pds_val)
        print("median", (dval1+dval2).min(dim=0)[0].median())
        maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
        maskValclose = maskValclose.cpu().numpy()
        maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
        maskValfar = maskValfar.cpu().numpy()
        # ardcaTrain, ardcaTest, ardcaVal, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
        # print("score", i)
        # print(i, ardcaTrain, ardcaTest, ardcaVal, ardcascoreH)
        
        train_iterator = DataLoader(pds_train, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        test_iterator = DataLoader(pds_test, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        val_iterator = DataLoader(pds_val, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        
        
        # Model hyperparameters
        src_vocab_size = 25#len(protein.vocab) 
        trg_vocab_size = 25#len(protein_trans.vocab) 
        embedding_size = 255#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
        src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
        src_position_embedding = PositionalEncoding(config.embedding_size, max_len=len_input,device=device)
        trg_position_embedding = PositionalEncoding(config.embedding_size, max_len=len_output, device=device)
                
        model = Transformer(
            config.embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            config.num_heads,
            config.num_layer,
            config.num_layer,
            config.forward_expansion,
            config.dropout,
            src_position_embedding,
            trg_position_embedding,
            device,
            onehot=onehot,
        ).to(device)
        
        #whyyy 'cpu?'
        
        
        step = 0
        step_ev = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        learning_rate = config.lr
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        if config.optim == "adam":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
        
        if config.optim == "sgd":
            optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
        
        
        
        pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
        for epoch in range(num_epochs+1):
            print(f"[Epoch {epoch} / {num_epochs}]")
            model.train()
            lossesCE = []
            accuracyTrain = 0
            for batch_idx, batch in enumerate(train_iterator):
                if config.precision == 'classic':
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
                if config.precision == 'mixed':
                    with torch.cuda.amp.autocast():
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
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                    scaler.step(optimizer)
                    scaler.update()
                
                
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
            wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal,  "accuracyVal":accuracyVal ,  "accuracyTrain": accuracyTrain, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
            if epoch%100==0:
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
                wandb.log({ "scoreMatching Val": scoreMatchingVal, "scoreMatchingValClose": scoreMatchingValClose, "scoreMatchingVal Far": scoreMatchingValFar,"epoch":epoch})
                # wandb.log({"scoreMatching Val": scoreMatching, "epoch":epoch})


sweep_id = wandb.sweep(sweep_config, project="Sweep embed other"+str(i))
wandb.agent(sweep_id, train)




















# def train(conf, nametosave):
#     i=46
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pathtoFolder = "/home/Datasets/DomainsInter/processed/"
#     count = 0
#     onehot=False
#     num_epochs = 3000
#     Unalign = False
#     # Model hyperparameters--> CAN BE CHANGED
#     pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
#     inputsize, outputsize = getLengthfromCSV(pathTofile)
#     os.path.isfile(pathTofile)
#     count +=1
#     print("ddi", i, " is running")
#     name = "combined_MSA_ddi_" +str(i)+"_joined"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     wandb.init(project="Hyperparamter Sweep embed 2", entity="barthelemymp")
#     config_dict = {
#       "num_layers": conf.num_layer,
#       "embedding":conf.embedding_size,
#       "forward_expansion": conf.forward_expansion,
#       "batch_size": conf.batch_size,
#       "Encoder": "Positional",
#       "Family":i,
#       "dropout":conf.dropout,
#       "len input":inputsize,
#       "len output":outputsize,
#       "scheduler": "none",
#       "loss": "CE",
#       "alphaParameter":0.0,
#       "weight_decay":conf.weight_decay
#     }
#     wandb.config.update(config_dict) 
#     #Dataset
#     train_path = pathtoFolder + name +'_train.csv'
#     val_path = pathtoFolder + name +'_val.csv'
#     test_path = pathtoFolder + name +'_test.csv'
#     #add 2 for start and end token 
#     len_input = inputsize + 2
#     len_output =outputsize + 2
#     pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
#     pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
#     pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
#     ntrain = len(pds_train)
#     nval = len(pds_val)
#     dval1,dval2 = distanceTrainVal(pds_train, pds_val)
#     print("median", (dval1+dval2).min(dim=0)[0].median())
#     maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
#     maskValclose = maskValclose.cpu().numpy()
#     maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
#     maskValfar = maskValfar.cpu().numpy()
#     train_iterator = DataLoader(pds_train, batch_size=conf.batch_size,
#                     shuffle=True, num_workers=0, collate_fn=default_collate)
#     test_iterator = DataLoader(pds_test, batch_size=conf.batch_size,
#                     shuffle=True, num_workers=0, collate_fn=default_collate)
#     val_iterator = DataLoader(pds_val, batch_size=conf.batch_size,
#                     shuffle=True, num_workers=0, collate_fn=default_collate)
#     # Model hyperparameters
#     src_vocab_size = 25#len(protein.vocab) 
#     trg_vocab_size = 25#len(protein_trans.vocab) 
#     #embedding_size = 255#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
#     src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
#     src_position_embedding = PositionalEncoding(conf.embedding_size, max_len=len_input,device=device)
#     trg_position_embedding = PositionalEncoding(conf.embedding_size, max_len=len_output, device=device)
#     model = Transformer(
#         conf.embedding_size,
#         src_vocab_size,
#         trg_vocab_size,
#         src_pad_idx,
#         conf.num_heads,
#         conf.num_layer,
#         conf.num_layer,
#         conf.forward_expansion,
#         conf.dropout,
#         src_position_embedding,
#         trg_position_embedding,
#         device,
#         onehot=onehot,
#     ).to(device)
#     step = 0
#     step_ev = 0
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     learning_rate = 3e-4
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_normal_(p)
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=conf.weight_decay)
#     pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
#     criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
#     for epoch in range(num_epochs+1):
#         print(f"[Epoch {epoch} / {num_epochs}]")
#         model.train()
#         lossesCE = []
#         accuracyTrain = 0
#         for batch_idx, batch in enumerate(train_iterator):
#             inp_data, target= batch[0], batch[1]
#             output = model(inp_data, target[:-1, :])
#             accuracyTrain += accuracy(batch, output, onehot=False).item()
#             output = output.reshape(-1, output.shape[2])#keep last dimension
            
#             if onehot:
#                 _, targets_Original = target.max(dim=2)
#             else:
#                 targets_Original= target
#             targets_Original = targets_Original[1:].reshape(-1)
#             optimizer.zero_grad()
#             loss = criterion(output, targets_Original)
#             lossesCE.append(loss.item())
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
#             optimizer.step()
#         mean_lossCETrain = sum(lossesCE) / len(lossesCE)
#         accuracyTrain = accuracyTrain/ntrain
#         # mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
#         step += 1
#         # scheduler.step(mean_lossCETrain)
#         model.eval()
#         lossesCE_eval = []
#         lossesMatching_eval = []
#         accuracyVal = 0
#         if epoch%1==0:
#             with  torch.no_grad():
#                 for batch_idx, batch in enumerate(val_iterator):
#                     inp_data, target= batch[0], batch[1]
#                     inp_data = inp_data.to(device)
#                     output = model(inp_data, target[:-1, :])
#                     accuracyVal += accuracy(batch, output, onehot=False).item()
#                     output = output.reshape(-1, output.shape[2]) #keep last dimension
#                     if onehot:
#                         _, targets_Original = target.max(dim=2)
#                     else:
#                         targets_Original= target
#                     targets_Original = targets_Original[1:].reshape(-1)
#                     loss_eval = criterion(output, targets_Original)
#                     lossesCE_eval.append(loss_eval.item()) 
#                 mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
#                 accuracyVal = accuracyVal/nval
#                 step_ev +=1
#         wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "accuracyVal":accuracyVal ,  "accuracyTrain": accuracyTrain, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
#         if epoch%200==0:
#             criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
#             model.eval()
#             criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
#             scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
#             if epoch == 400:
#                 nnn = nametosave + str(400)+ ".npz"
#                 np.save(nnn, scoreHungarianVal)
#             if epoch == 3000:
#                 nnn = nametosave + str(3000)+ ".npz"
#                 np.save(nnn, scoreHungarianVal)
#             scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
#             scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
#             scoreMatchingValClose = sum((scoHVal[0]==scoHVal[1])[maskValclose])
#             scoreMatchingValFar = sum((scoHVal[0]==scoHVal[1])[maskValfar])
#             # scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
#             # scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
#             # scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
#             wandb.log({ "scoreMatching Val": scoreMatchingVal, "scoreMatchingValClose": scoreMatchingValClose, "scoreMatchingVal Far": scoreMatchingValFar,"epoch":epoch})
#             # wandb.log({"scoreMatching Val": scoreMatching, "epoch":epoch})
#     wandb.finish()
    
    
# class myobj(object):
#     pass


# conf = myobj()
# conf.forward_expansion = 512
# conf.num_heads = 5
# conf.batch_size = 32
# conf.num_layer = 2
# conf.embedding_size  = 55
# conf.dropout  = 0.1
# conf.weight_decay  = 0.001     
# train(conf, "scoreH_nonoverfit")

# conf = myobj()
# conf.forward_expansion = 512
# conf.num_heads = 5
# conf.batch_size = 32
# conf.num_layer = 6
# conf.embedding_size  = 55
# conf.dropout  = 0.1
# conf.weight_decay  = 0.001     
# train(conf, "scoreH_overfit")

# 12+4