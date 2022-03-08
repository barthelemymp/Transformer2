# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:24:01 2022

@author: bartm
"""
import subprocess
import matplotlib.pyplot as plt





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
sys.path.append("")
from ProteinTransformer import *
from ProteinsDataset import *
from MatchingLoss import *
from utils import *
from ardca import *

def buildhmm(hmmout, ali):
    subprocess.run(["hmmbuild", "--symfrac","0.0", hmmout, ali])




def getlists(df, fam):
    pdblist = list(df[df["id"]==fam]["pdb"])
    chain1list = list(df[df["id"]==fam]["chain1"])
    chain2list = list(df[df["id"]==fam]["chain2"])
    return pdblist, chain1list, chain2list



family_list = [ 69, 71,157,160,251, 258, 97,103,181, 192, 197,303,304,308,358,504, 634, 815, 972, 972, 980, 1208, 1213, 1214, 17, 46,132] 
pdbtracker = pd.read_csv("pdbtracker.csv")



print("import done")
#torch.functional.one_hot
pathtoFolder = "/home/feinauer/Datasets/DomainsInter/processed/"
torch.set_num_threads(4)
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 1
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 25#len(protein.vocab) 
trg_vocab_size = 25#len(protein_trans.vocab) 
embedding_size = 55#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
#EPOCHS 
num_epochs =1000
Unalign = False
onehot=False
wd=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#             ilist.append(i)
            


onehot=False

for i in family_list:
    
    modelpath = "TransSimple_fam"+str(i)+".pth.tar"
    
    pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
    inputsize, outputsize = getLengthfromCSV(pathTofile)
    os.path.isfile(pathTofile)
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    learning_rate = 5e-5
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
            
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    load_checkpoint(torch.load(modelpath), model, optimizer)


    ### Create tempfiles
    tempFile=next(tempfile._get_candidate_names())+".npy"
    mode = "inter"
    #### getlist
    pdbtracker = pd.read_csv("pdbtracker.csv")
    pdblist, chain1list, chain2list = getlists(pdbtracker, i)
    np.save("pdblisttemp.npy", pdblist)
    np.save("chain1listtemp.npy", chain1list)
    np.save("chain2listtemp.npy", chain2list)
    hmmRadical =pathtoFolder+"hmm_"+str(i)+"_"
    tempTrainr = writefastafrompds(pds_train)
    tempTrain=tempTrainr+"joined.faa"
    output = subprocess.check_output(["stdbuf", "-oL", "julia", "contactPlot_merged.jl", tempTrain, "pdblisttemp.npy", "chain1listtemp.npy", "chain2listtemp.npy", hmmRadical, tempFile, mode])
    print(output)
    ppvO = np.load(tempFile)
    removetemp(tempTrainr)
    
    ### sample times 2
    max_len = len_output
    pds_sample = copy.deepcopy(pds_train)
    batchIndex = makebatchList(len(pds_sample), 300)
    for batchI in batchIndex:
        sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
        # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
        pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
        pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
    tempTrainr = writefastafrompds(pds_sample)
    tempTrain=tempTrainr+"joined.faa"
    output = subprocess.check_output(["julia", "contactPlot_merged.jl", tempTrain, "pdblisttemp.npy", "chain1listtemp.npy", "chain2listtemp.npy", hmmRadical, tempFile, mode])
    print(output)
    ppvS1 = np.load(tempFile)
    removetemp(tempTrainr)
    
        
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
    output = subprocess.check_output(["julia", "contactPlot_merged.jl", tempTrain, "pdblisttemp.npy", "chain1listtemp.npy", "chain2listtemp.npy", hmmRadical, tempFile, mode])
    print(output)
    ppvS3 = np.load(tempFile)
    removetemp(tempTrainr)
        
        
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
    for batchI in batchIndex:
        sampled = model.sample(pds_sample[batchI][0], max_len, nsample=1, method="simple")
        # pds_sample.tensorOUT[:,batchI]=sampled.max(dim=2)[1]
        pds_sample.tensorOUT=torch.cat([pds_sample.tensorOUT,sampled.max(dim=2)[1] ],dim=1)
        pds_sample.tensorIN=torch.cat([pds_sample.tensorIN,pds_sample.tensorIN[:,batchI] ], dim=1)
    tempTrainr = writefastafrompds(pds_sample)
    tempTrain=tempTrainr+"joined.faa"
    output = subprocess.check_output(["julia", "contactPlot_merged.jl", tempTrain, "pdblisttemp.npy", "chain1listtemp.npy", "chain2listtemp.npy", hmmRadical, tempFile, mode])
    print(output)
    ppvS8 = np.load(tempFile)
    x = np.array(range(1,len(ppvO)+1))
    plt.plot(x,ppvO, label="Original")
    plt.plot(x,ppvS1, label="sampled*1")
    plt.plot(x,ppvS3, label="sampled*3")
    plt.plot(x,ppvS8, label="sampled*8")
    plt.title("PPV" + str(i))
    plt.legend()
    plt.xscale("log")
    plt.savefig("ppvS_"+str(i)+".png")
    plt.clf()




    


    


    



# for i in range(7000):
#     original_pathj =  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_"+str(i)+"_joined.fasta" 
#     original_path1 =  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_"+str(i)+"_1.fasta" 
#     original_path2 =  "/home/Datasets/DomainsInter/processed/combined_MSA_ddi_"+str(i)+"_2.fasta" 
#     if os.path.isfile(original_pathj):
#         print(i)
#         subprocess.run(["hmmbuild", "--symfrac", "0.0", "hmm_"+str(i)+"_joined.hmm", original_pathj])
#         subprocess.run(["hmmbuild", "--symfrac", "0.0", "hmm_"+str(i)+"_1.hmm", original_path1])
#         subprocess.run(["hmmbuild", "--symfrac",  "0.0", "hmm_"+str(i)+"_2.hmm", original_path2])











