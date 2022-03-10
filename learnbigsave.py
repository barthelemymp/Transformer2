import argparse
import numpy as np
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
torch.set_num_threads(8)


# Params

import sys
family = str(sys.argv[1])
i = int(family)

nlayer = 2
num_encoder_layers = nlayer
num_decoder_layers = nlayer
embedding_size = 55
if embedding_size>0:
    onehot=False
Unalign = False
num_heads = 1
batch_size = 32
forward_expansion = 2048
num_epochs= 5000
src_vocab_size = 25
trg_vocab_size = 25
dropout = 0.10
wd = 0.5
alpha = 0.0
##### Training simple 
pathtoFolder = "/home/feinauer/Datasets/DomainsInter/processed/"
pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
inputsize, outputsize = getLengthfromCSV(pathTofile)

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
downsampleslist = [100, 500, 1000, 3000, len(pds_train)]
# ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
# print("score", i)
# print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)
for downsamples in downsampleslist:
    pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
    pds_train.downsample(downsamples)


    
    
    
    
    
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
    wandb.init(project="Compare reg", entity="barthelemymp")
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
      "wd":wd,
      "downsamples": downsamples,
    }
    wandb.config.update(config_dict) 
    
    tempFile=next(tempfile._get_candidate_names())+".npy"
    mode = "inter"
    #### getlist
    pdbtracker = pd.read_csv("pdbtracker.csv")
    pdblist, chain1list, chain2list = getlists(pdbtracker, i)
    np.save("pdblisttemph"+str(i)+".npy", pdblist)
    np.save("chain1listtemph"+str(i)+".npy", chain1list)
    np.save("chain2listtemph"+str(i)+".npy", chain2list)
    hmmRadical =pathtoFolder+"hmm_"+str(i)+"_"
    tempTrainr = writefastafrompds(pds_train)
    tempTrain=tempTrainr+"joined.faa"
    output = subprocess.check_output(["stdbuf", "-oL", "julia", "contactPlot_merged.jl", tempTrain, "pdblisttemph"+str(i)+".npy", "chain1listtemph"+str(i)+".npy", "chain2listtemph"+str(i)+".npy", hmmRadical, tempFile, mode])
    print(output)
    ppvO = np.load(tempFile)
    x_values = np.array(range(1,len(ppvO)+1))
    data = [[x, y] for (x, y) in zip(x_values, ppvO)]
    table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"PPV original" : wandb.plot.line(table, "x", "y",
               title="PPV original")})
    
    step = 0
    step_ev = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    learning_rate = 5e-5
    
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
            wandb.log({ "scoreMatching Val": scoreMatchingVal, "scoreMatchingValClose": scoreMatchingValClose, "scoreMatchingVal Far": scoreMatchingValFar,"epoch":epoch})
        if epoch%1000==0:

            max_len = len_output
            pds_sample = copy.deepcopy(pds_train)
            batchIndex = makebatchList(len(pds_sample), 300)
            for batchI in batchIndex:
                sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
                # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
                pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
                pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
            for batchI in batchIndex:
                sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
                # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
                pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
                pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
            for batchI in batchIndex:
                sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
                # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
                pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
                pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
            for batchI in batchIndex:
                sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
                # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
                pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
                pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
                
                
            tempTrainr = writefastafrompds(pds_sample)
            tempTrain=tempTrainr+"joined.faa"
            output = subprocess.check_output(["julia", "contactPlot_merged.jl", tempTrain, "pdblisttemph"+str(i)+".npy", "chain1listtemph"+str(i)+".npy", "chain2listtemph"+str(i)+".npy", hmmRadical, tempFile, mode])
            print(output)
            ppv = np.load(tempFile)
            x_values = np.array(range(1,len(ppv)+1))
            data = [[x, y] for (x, y) in zip(x_values, ppv)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({"PPV"+str(epoch) : wandb.plot.line(table, "x", "y",
                       title="Custom Y vs X Line Plot"), "epoch":epoch})
    
        
    
    wandb.finish()







# alpha = -0.5
# wd = 0.0
# ##### Training simple 
# pathtoFolder = "/home/feinauer/Datasets/DomainsInter/processed/"
# pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
# inputsize, outputsize = getLengthfromCSV(pathTofile)

# print("ddi", i, " is running")
# name = "combined_MSA_ddi_" +str(i)+"_joined"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #Dataset
# train_path = pathtoFolder + name +'_train.csv'
# val_path = pathtoFolder + name +'_val.csv'
# test_path = pathtoFolder + name +'_test.csv'
# #add 2 for start and end token 
# len_input = inputsize + 2
# len_output =outputsize + 2
# pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
# ntrain = len(pds_train)
# nval = len(pds_val)
# ntest = len(pds_test)
# dval1,dval2 = distanceTrainVal(pds_train, pds_val)
# print("median", (dval1+dval2).min(dim=0)[0].median())
# maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
# maskValclose = maskValclose.cpu().numpy()
# maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
# maskValfar = maskValfar.cpu().numpy()
# # ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
# # print("score", i)
# # print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

# train_iterator = DataLoader(pds_train, batch_size=batch_size,
#                 shuffle=True, num_workers=0, collate_fn=default_collate)
# test_iterator = DataLoader(pds_test, batch_size=batch_size,
#                 shuffle=True, num_workers=0, collate_fn=default_collate)
# val_iterator = DataLoader(pds_val, batch_size=batch_size,
#                 shuffle=True, num_workers=0, collate_fn=default_collate)

# # Model hyperparameters

# src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
# src_position_embedding = PositionalEncoding(embedding_size, max_len=len_input,device=device)
# trg_position_embedding = PositionalEncoding(embedding_size, max_len=len_output, device=device)
        
# model = Transformer(
#     embedding_size,
#     src_vocab_size,
#     trg_vocab_size,
#     src_pad_idx,
#     num_heads,
#     num_encoder_layers,
#     num_decoder_layers,
#     forward_expansion,
#     dropout,
#     src_position_embedding,
#     trg_position_embedding,
#     device,
#     onehot=onehot,
# ).to(device)

# #whyyy 'cpu?'
# wandb.init(project="Compare reg", entity="barthelemymp")
# config_dict = {
#   "num_layers": num_encoder_layers,
#   "embedding":embedding_size,
#   "forward_expansion": forward_expansion,
#   "batch_size": batch_size,
#   "Encoder": "Positional",
#   "Family":i,
#   "dropout":dropout,
#   "len input":len_input,
#   "len output":len_output,
#   "sizetrain": len(pds_train),
#   "sizeval": len(pds_val),
#   "num_heads": num_heads,
#   "loss": "CE",
#   "wd":wd,
#   "alpha" = alpha,
# }
# wandb.config.update(config_dict) 

# step = 0
# step_ev = 0
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# learning_rate = 5e-5

# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_normal_(p)
        
        
    
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
# pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
# criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])
# criterion_raw = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
# criterionMatching = nn.CrossEntropyLoss()

# for epoch in range(num_epochs+1):
#     print(f"[Epoch {epoch} / {num_epochs}]")
#     model.train()
#     lossesCE = []
#     accuracyTrain = 0
#     for batch_idx, batch in enumerate(train_iterator):
#         optimizer.zero_grad()
#         # sparseoptim.zero_grad()
#         # opt_sparse.zero_grad()
#         # opt_dense.zero_grad()
#         # lossCE, lossEntropy, acc = ConditionalSquaredEntropyMatchingLoss(batch, model, criterion, device, samplingMultiple=10, gumbel=gumbel)
#         lossCE, lossEntropy = SamplerContrastiveMatchingLoss(batch, model,
#                                             criterion_raw,
#                                             criterionMatching,
#                                             device,
#                                             accumulate=False,
#                                             alpha=alpha,
#                                             numberContrastive=10,
#                                             sampler="gumbel")
        
#         #accuracyTrain += acc
#         lossesCE.append(lossCE.item())
#         loss = lossCE + alpha * lossEntropy
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
#         # opt_sparse.step()
#         # opt_dense.step()
#         # sparseoptim.step()
#         optimizer.step()
            
        
        
#     mean_lossCETrain = sum(lossesCE) / len(lossesCE)
#     accuracyTrain = accuracyTrain/ntrain
#     # mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
#     step += 1
#     # scheduler.step(mean_lossCETrain)
#     model.eval()
#     lossesCE_eval = []
#     lossesMatching_eval = []
#     accuracyVal = 0
    
#     if epoch%1==0:
#         with  torch.no_grad():
#             for batch_idx, batch in enumerate(val_iterator):
#                 inp_data, target= batch[0], batch[1]
#                 inp_data = inp_data.to(device)
#                 output = model(inp_data, target[:-1, :])
#                 accuracyVal += accuracy(batch, output, onehot=False).item()
#                 output = output.reshape(-1, output.shape[2]) #keep last dimension
#                 if onehot:
#                     _, targets_Original = target.max(dim=2)
#                 else:
#                     targets_Original= target
#                 targets_Original = targets_Original[1:].reshape(-1)
#                 loss_eval = criterion(output, targets_Original)
#                 lossesCE_eval.append(loss_eval.item()) 
#             mean_lossVal = sum(lossesCE_eval) / len(lossesCE_eval)
#             accuracyVal = accuracyVal/nval
#             step_ev +=1
#     #wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal,  "accuracyVal":accuracyVal ,  "accuracyTrain": accuracyTrain, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
    
    
#     lossesCE_test = []
#     lossesMatching_test = []
#     accuracytest = 0
    
#     if epoch%1==0:
#         with  torch.no_grad():
#             for batch_idx, batch in enumerate(test_iterator):
#                 inp_data, target= batch[0], batch[1]
#                 inp_data = inp_data.to(device)
#                 output = model(inp_data, target[:-1, :])
#                 accuracytest += accuracy(batch, output, onehot=False).item()
#                 output = output.reshape(-1, output.shape[2]) #keep last dimension
#                 if onehot:
#                     _, targets_Original = target.max(dim=2)
#                 else:
#                     targets_Original= target
#                 targets_Original = targets_Original[1:].reshape(-1)
#                 loss_test = criterion(output, targets_Original)
#                 lossesCE_test.append(loss_test.item()) 
#             mean_losstest = sum(lossesCE_test) / len(lossesCE_test)
#             accuracytest = accuracytest/ntest
#             step_ev +=1
#     wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossVal, "test loss CE": mean_losstest,  "accuracyVal":accuracyVal , "accuracytest":accuracytest ,  "accuracyTrain": accuracyTrain, "epoch":epoch})
    
    
#     if epoch%200==0:
#         criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
#         model.eval()
#         criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
#         scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
#         scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
#         scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
#         scoreMatchingValClose = sum((scoHVal[0]==scoHVal[1])[maskValclose])
#         scoreMatchingValFar = sum((scoHVal[0]==scoHVal[1])[maskValfar])
#         # scoreHungarianTrain = HungarianMatchingBS(pds_train, model,100)
#         # scoHTrain = scipy.optimize.linear_sum_assignment(scoreHungarianTrain)
#         # scoreMatchingTrain = sum(scoHTrain[0]==scoHTrain[1])
#         wandb.log({ "scoreMatching Val": scoreMatchingVal, "scoreMatchingValClose": scoreMatchingValClose, "scoreMatchingVal Far": scoreMatchingValFar,"epoch":epoch})
        

    

# wandb.finish()














