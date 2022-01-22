# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:10:31 2022

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
onehot =False
dataset_list_train = []
dataset_list_test = []
dataset_list_val = []

for i in ilist:
    pathTofile = pathtoFolder+ "PPI_" +str(i)+"_joined.csv"
    inputsize, outputsize = getLengthfromCSV(pathTofile)
    os.path.isfile(pathTofile)
    count +=1
    print("ddi", i, " is running")
    name = "PPI_" +str(i)+"_joined"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Dataset
    train_path = pathtoFolder + name +'_train.csv'
    val_path = pathtoFolder + name +'_val.csv'
    test_path = pathtoFolder + name +'_test.csv'
    
    #add 2 for start and end token 
    len_input = inputsize + 2
    len_output =outputsize + 2
    try:
        splitcsv(pathtoFolder, name, repartition, shuffle=True, maxval=500)
        pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
    except:
        splitcsv(pathtoFolder, name, repartition, shuffle=True, maxval=500)
        pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
        pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign,filteringOption='and', returnIndex=True,onehot=onehot)
    print(len(pds_val))
    
    # train_iterator = DataLoader(pds_train, batch_size=batch_size,
    #                 shuffle=True, num_workers=0, collate_fn=default_collate)
    # test_iterator = DataLoader(pds_test, batch_size=batch_size,
    #                 shuffle=True, num_workers=0, collate_fn=default_collate)
    # val_iterator = DataLoader(pds_val, batch_size=batch_size,
    #                 shuffle=True, num_workers=0, collate_fn=default_collate)
    dataset_list_train.append(pds_train)
    dataset_list_test.append(pds_test)
    dataset_list_val.append(pds_val)
    
    
pds = dataset_list_train[0]

for i in range(1,len(dataset_list_train)):
    print(i)
    pds.join(dataset_list_train[i])
        