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
print("import done")
#torch.functional.one_hot
#pathtoFolder = "/home/bart/Datasets/DomainsInter/processed/"
pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 5
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.10
forward_expansion = 1024
repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.1, 0.5]
wd_list = [0.0]#, 0.00005]
ilist = [46, 69, 71,157,160,251, 258]
for wd in wd_list:
    for i in ilist:
        # i=46
        # wd =0.0
        alpha = 0.0
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
        
        #whyyy 'cpu?'
        wandb.init(project="Finetuning PMI", entity="barthelemymp")
        config_dict = {
          "num_layers": num_encoder_layers,
          "forward_expansion": forward_expansion,
          "batch_size": batch_size,
          "Encoder": "Positional",
          "Family":i,
          "dropout":dropout,
          "len input":len_input,
          "len output":len_output,
          "scheduler": "none",
          "loss": "CE",
          "alphaParameter":0.0,
          "weight_decay":wd
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
        criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
        for epoch in range(num_epochs+1):
            print(f"[Epoch {epoch} / {num_epochs}]")
            model.train()
            lossesCE = []
            for batch_idx, batch in enumerate(train_iterator):
                inp_data, target= batch[0], batch[1]
                output = model(inp_data, target[:-1, :])
                output = output.reshape(-1, output.shape[2])#keep last dimension
                _, targets_Original = target.max(dim=2)
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
                        _, targets_Original = target.max(dim=2)
                        targets_Original = targets_Original[1:].reshape(-1)
                        loss_eval = criterion(output, targets_Original)
                        lossesCE_eval.append(loss_eval.item()) 
                    mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
                    step_ev +=1
            wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "alpha":alpha, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
            if epoch%200==0:
                criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
                model.eval()
                criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
                scoreHungarianVal = HungarianMatching(pds_val, model)
                scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
                scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
                scoreHungarianTrain = HungarianMatching(pds_train, model)
                scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
                scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
                wandb.log({"scoreMatching Train": scoreMatchingTrain, "scoreMatching Val": scoreMatchingVal, "epoch":epoch})
                # wandb.log({"scoreMatching Val": scoreMatching, "epoch":epoch})
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename="trans"+str(i)+"_wd"+str(wd)+"_simple_epoch"+str(epoch)+"_"+str(num_encoder_layers)+"_"+str(forward_expansion)+".pth.tar")
        wandb.finish()
 
        ########  Start Finetuning

        num_epochs_finetuning = 301
        accumulate = True
        for alpha in alphalist:
            wandb.init(project="Finetuning PMI", entity="barthelemymp")
            config_dict = {
              "num_layers": num_encoder_layers,
              "forward_expansion": forward_expansion,
              "batch_size": batch_size,
              "Encoder": "Positional",
              "Family":i,
              "dropout":dropout,
              "len input":len_input,
              "len output":len_output,
              "scheduler": "none",
              "alphaParameter":alpha,
              "weight_decay": wd,
              "loss": "CE+PMI"
            }
            wandb.config.update(config_dict) 
            load_checkpoint(torch.load("trans"+str(i)+"_wd"+str(wd)+"_simple_epoch"+str(5000)+"_"+str(num_encoder_layers)+"_"+str(forward_expansion)+".pth.tar"), model, optimizer)
            step = 0
            step_ev = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            pad_idx = "<pad>"
            criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
            criterionMatching = nn.CrossEntropyLoss()
            for epoch in range(num_epochs_finetuning):
                print(f"[Epoch {epoch} / {num_epochs}]")
                if epoch%10==0:
                    model.eval()
                    criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
                    scoreHungarianVal = HungarianMatching(pds_val, model)
                    scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
                    scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
                    scoreHungarianTrain = HungarianMatching(pds_train, model)
                    scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
                    scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
                    wandb.log({"scoreMatching Train": scoreMatchingTrain, "scoreMatching Val": scoreMatchingVal, "epoch":epoch})
                model.train()
                lossesCE = []
                lossesMatching = []
                for batch_idx, batch in enumerate(train_iterator):
                    lossCE, lossMatching = PointwiseMutualInformationLoss(batch,
                                                                    model,
                                                                    scoreHungarianTrain,
                                                                    pds_train,
                                                                    criterion,
                                                                    criterionMatching,
                                                                    device,
                                                                    accumulate=accumulate,
                                                                    alpha=alpha,
                                                                    numberContrastive=10)
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
                model.eval()
                lossesCE_eval = []
                lossesMatching_eval = []
                if epoch%1==0:
                    with  torch.no_grad():
                        for batch_idx, batch in enumerate(val_iterator):
                            lossCE, lossMatching = PointwiseMutualInformationLoss(batch,
                                                                            model,
                                                                            scoreHungarianVal,
                                                                            pds_val,
                                                                            criterion,
                                                                            criterionMatching,
                                                                            device,
                                                                            alpha=alpha)
                            lossesCE_eval.append(lossCE.item())
                            lossesMatching_eval.append(lossMatching.item())
                        mean_lossCEVal = sum(lossesCE_eval) / len(lossesCE_eval)
                        mean_lossMatchingVal = sum(lossesMatching_eval) / len(lossesMatching_eval)
                        step_ev +=1
                wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossCEVal,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha, "Train loss Matching": mean_lossMatchingTrain, "epoch":epoch})
            wandb.finish()
