# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:51:27 2022

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
#pathtoFolder = "/home/bart/Datasets/DomainsInter/processed/"
pathtoFolder = "/home/Datasets/DomainsInter/PPIprocessed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 5
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.1]
wd_list = [0.0]#, 0.00005]
ilist = [1,22,3,5,7,8,9,10,12,16,19,21,2,27,31]
for alpha in alphalist:
    
    for i in ilist:
        # i=46
        # wd =0.0
        wd = 0.0
        ##### Training simple 
        
        pathTofile = pathtoFolder+ "PPI_" +str(i)+"_joined.csv"
        
        inputsize, outputsize = getLengthfromCSV(pathTofile)
        os.path.isfile(pathTofile)
        count +=1
        print("ddi", i, " is running")
        name = "PPI_" +str(i)+"_joined"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #splitcsv(pathtoFolder, name, repartition, shuffle=True)
        #Dataset
        train_path = pathtoFolder + name +'_train.csv'
        val_path = pathtoFolder + name +'_val.csv'
        test_path = pathtoFolder + name +'_test.csv'
        
        #add 2 for start and end token 
        len_input = inputsize + 2
        len_output =outputsize + 2
        
        pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True)
        pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True)
        pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True)

        train_iterator = DataLoader(pds_train, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        test_iterator = DataLoader(pds_test, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        val_iterator = DataLoader(pds_val, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        
        
        # Model hyperparameters
        src_vocab_size = 25#len(protein.vocab) 
        trg_vocab_size = 25#len(protein_trans.vocab) 
        embedding_size = 25#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
        src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
        src_position_embedding = PositionalEncoding(embedding_size, max_len=len_input,device=device)
        trg_position_embedding = PositionalEncoding(embedding_size, max_len=len_output, device=device)

        accumulate = True
        for alpha in alphalist:
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
                device
            ).to(device)
            
            
            wandb.init(project="PPI sampler contrastive", entity="barthelemymp")
            config_dict = {
              "num_layers": num_encoder_layers,
              "forward_expansion": forward_expansion,
              "batch_size": batch_size,
              "Encoder": "Positional",
              "dataset":"PPI",
              "Family":i,
              "dropout":dropout,
              "len input":len_input,
              "len output":len_output,
              "trainsize":len(pds_train),
              "valsize":len(pds_val),
              "scheduler": "none",
              "alphaParameter":alpha,
              "weight_decay": wd,
              "loss": "CE+Simple sampler CEContrastive"
            }
            wandb.config.update(config_dict) 
            
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

        
            step = 0
            step_ev = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            pad_idx = "<pad>"
            criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
            criterionMatching = nn.CrossEntropyLoss()
            criterionVal = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
            for epoch in range(num_epochs):
                print(f"[Epoch {epoch} / {num_epochs}]")
                if epoch%1000==999:
                    model.eval()
                    criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
                    scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
                    scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
                    scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
                    scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
                    scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
                    scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
                    wandb.log({"scoreMatching Train": scoreMatchingTrain, "scoreMatching Val": scoreMatchingVal, "epoch":epoch})
                    
                model.train()
                lossesCE = []
                lossesMatching = []
                for batch_idx, batch in enumerate(train_iterator):

                    
                    lossCE, lossMatching = SamplerContrastiveMatchingLoss(batch,
                                                        model,
                                                        pds_train,
                                                        criterion,
                                                        criterionMatching,
                                                        device,
                                                        accumulate=accumulate,
                                                        alpha=alpha,
                                                        numberContrastive=10,
                                                        sampler="simple")
                    
                    lossesCE.append(lossCE.item())
                    lossesMatching.append(lossMatching.item())
                    if accumulate ==False:
                        loss = lossCE + alpha*lossMatching
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                    # Gradient descent step
                    optimizer.step()
                    optimizer.zero_grad()
                mean_lossCETrain = sum(lossesCE) / len(lossesCE)
                mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
                # writer.add_scalar("Training loss", mean_loss, global_step=step)
                # step += 1
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
                            _, targets_Original = target.max(dim=2)
                            targets_Original = targets_Original[1:].reshape(-1)
                            loss_eval = criterionVal(output, targets_Original)
                            
                            lossesCE_eval.append(loss_eval.item())

                        mean_lossCEVal = sum(lossesCE_eval) / len(lossesCE_eval)

                        step_ev +=1
                wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossCEVal, "alpha":alpha, "Train loss Matching": mean_lossMatchingTrain, "epoch":epoch})
            wandb.finish()
