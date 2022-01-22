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
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.01, 0.1]
wd_list = [0.0]#, 0.00005]
ilist = [17, 46, 69, 71,157,160,251, 258]
pds_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(18,50):
    print(i)
    pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
    if os.path.isfile(pathTofile)==False:
        continue
    try:
        inputsize, outputsize = getLengthfromCSV(pathTofile)
        pds_train = ProteinTranslationDataset(pathTofile, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True)
        pds_list.append(pds_train)
        print(i, inputsize, outputsize, len(pds_train))
    except:
        continue
    
   
pds_list[0].join(pds_list[1]) 
   
for i in range(1,5):
    print(i)
    pds_list[0].join(pds_list[i])
    
    
    
    
    