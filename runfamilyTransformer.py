# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:54:11 2021

@author: bartm
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:27:32 2021

@author: bartm
"""


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
from utils import *

import sys
family = str(sys.argv[1])
i = int(family)
i=17




count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 10
num_heads = 5
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.10
forward_expansion = 2048
# repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs = 1400
Unalign = False







# pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
inputsize, outputsize = getLengthfromCSV("test_real.csv")# pathTofile)
# os.path.isfile(pathTofile)
# count +=1
# print("ddi", i, " is running")
# name = "combined_MSA_ddi_" +str(i)+"_joined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Dataset
train_path = "train_real.csv"# pathtoFolder + name +'_train.csv'
val_path = "val_real.csv"#pathtoFolder + name +'_val.csv'
test_path = "test_real.csv"#pathtoFolder + name +'_test.csv'

#add 2 for start and end token 
len_input = inputsize + 2
len_output =outputsize + 2

pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign)
# pds_train.tensorIN=pds_train.tensorIN[:,torch.randperm(pds_train.tensorIN.size()[1]),:]
# pds_train.tensorOUT=pds_train.tensorOUT[:,torch.randperm(pds_train.tensorOUT.size()[1]),:]
pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign)
# pds_test.tensorIN=pds_test.tensorIN[:,torch.randperm(pds_test.tensorIN.size()[1]),:]
# pds_test.tensorOUT=pds_test.tensorOUT[:,torch.randperm(pds_test.tensorOUT.size()[1]),:]
pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign)
# pds_val.tensorIN=pds_val.tensorIN[:,torch.randperm(pds_val.tensorIN.size()[1]),:]
# pds_val.tensorOUT=pds_val.tensorOUT[:,torch.randperm(pds_val.tensorOUT.size()[1]),:]

train_iterator = DataLoader(pds_train, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
test_iterator = DataLoader(pds_test, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
val_iterator = DataLoader(pds_val, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)


# Model hyperparameters
src_vocab_size = 25#len(protein.vocab) 
trg_vocab_size = 25#len(protein_trans.vocab) 

# assert src_vocab_size==trg_vocab_size, "the input vocabulary differs from output voc"
# assert src_vocab_size<=25, "Something wrong with length of vocab in input s."
# assert trg_vocab_size<=25, "Something wrong with length of vocab in output s."

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
wandb.init(project="Sandbox", entity="barthelemymp")
config_dict = {
  "num_layers": num_encoder_layers,
  "forward_expansion": forward_expansion,
  "batch_size": batch_size,
  "Encoder": "Positional",
  "Family":"HK-RR",
  "len input":len_input,
  "len output":len_output,
  "shuffled input output": "False",
  "Unalign": Unalign,
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

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

# The following line defines the pad token that is not taken into 
# consideration in the computation of the loss - cross entropy
pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])

load_model= False
save_model= True
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        if epoch%100 == 99:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # save_checkpoint(checkpoint, filename="fam"+str(i)+"Unalign.pth.tar")
            save_checkpoint(checkpoint, filename="famHkRR_pos_O.pth.tar")
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])#keep last dimension
        _, targets_Original = target.max(dim=2)
        targets_Original = targets_Original[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, targets_Original)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()
    mean_lossTrain = sum(losses) / len(losses)
    step += 1
    scheduler.step(mean_lossTrain)
    model.eval()
    losses_eval = []
    if epoch%1==0:
        with  torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                inp_data, target= batch[0], batch[1]
                print(inp_data.device)
                inp_data = inp_data.to(device)
                print(inp_data.device)
                output = model(inp_data, target[:-1, :])
                output = output.reshape(-1, output.shape[2]) #keep last dimension
                _, targets_Original = target.max(dim=2)
                targets_Original = targets_Original[1:].reshape(-1)
                loss_eval = criterion(output, targets_Original)
                losses_eval.append(loss_eval.item()) 
            mean_lossVal = sum(losses_eval) / len(losses_eval)
            step_ev +=1
    
    wandb.log({"Train loss": mean_lossTrain, "Val Loss":mean_lossVal,  "epoch":epoch})

