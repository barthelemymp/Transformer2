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
from DCA import *
torch.set_num_threads(8)

save_model=True
load_model=True
# Params

import sys
from copy import deepcopy
i = 17

onehot=False
Unalign = False
num_heads = 1
batch_size = 32
forward_expansion = 2048
num_epochs= 4000
src_vocab_size = 25
trg_vocab_size = 25
dropout = 0.10
wd = 0.0
##### Training simple 
#pathtoFolder = "/Data/DomainsInter/processed/"#turin
#pathtoFolder = "/home/meynard/Datasets/DomainsInter/processed/" ##Jussieu GPU
pathtoFolder = "/home/Datasets/DomainsInter/processed/"

wd = 0.0
##### Training simple 
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


for sizecut in [20, 40, 60, 80, 100]:
    print(sizecut)
    pds_traincut = deepcopy(pds_train)
    pds_traincut.tensorIN = pds_traincut.tensorIN[:sizecut,:]
    pds_traincut.tensorOUT = pds_traincut.tensorOUT[:sizecut,:]
    pds_valcut = deepcopy(pds_val)
    pds_valcut.tensorIN = pds_valcut.tensorIN[:sizecut,:]
    pds_valcut.tensorOUT = pds_valcut.tensorOUT[:sizecut,:]
    output = ARDCA_timeit(pds_traincut, pds_valcut)
    print(output)


