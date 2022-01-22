# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:50:51 2021

@author: bartm
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:56:22 2021

@author: bartm
"""


import sys
family = str("17")
i = int(family)

import wandb
wandb.login()

batch_size = 10
num_heads = 5
num_encoder_layers = 4
num_decoder_layers = 4
dropout = 0.10
forward_expansion = 4096

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math

sys.path.append("")
# from model_Trans_OneHot import * 
from ProteinTransformer import *
from utils import *

pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED

#EPOCHS 
num_epochs = 1000
Unalign = False


pathTofile = pathtoFolder+ "combined_MSA_ddi_" + family +"_joined.csv"
# =============================================================================
# if os.path.isfile(pathTofile)==False:
# continue
inputsize, outputsize = getLengthfromCSV(pathTofile)

# =============================================================================
count +=1
print("ddi", i, " is running")
name = "combined_MSA_ddi_" +str(i)+"_joined"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Dataset
train_path = pathtoFolder + name +'_train.csv'
val_path = pathtoFolder + name +'_val.csv'
test_path = pathtoFolder + name +'_test.csv'

len_input = inputsize + 2
len_output =outputsize + 2

pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign)
pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign)
pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign)
train_iterator = DataLoader(pds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=default_collate)
test_iterator = DataLoader(pds_test, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=default_collate)
val_iterator = DataLoader(pds_val, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=default_collate)
src_vocab_size = 25#len(protein.vocab) 
trg_vocab_size = 25#len(protein_trans.vocab) 
if src_vocab_size!=trg_vocab_size:
    print("the input vocabulary differs from output voc", src_vocab_size, trg_vocab_size)

assert src_vocab_size==trg_vocab_size, "the input vocabulary differs from output voc"
assert src_vocab_size<=25, "Something wrong with length of vocab in input s."
assert trg_vocab_size<=25, "Something wrong with length of vocab in output s."


embedding_size = 25
src_pad_idx = pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 

### Create the Sytructural Encoder
fastapath_src = pathtoFolder + "combined_MSA_ddi_17_joined_train_1.faa"
fastapath_trg = pathtoFolder + "combined_MSA_ddi_17_joined_train_2.faa"
pdbPath_src = "1BK0.pdb"
pdbPath_trg = "1BK0.pdb"
chain_src = "A"
chain_trg = "A"
src_position_embedding_Struct = StructuralAlignedEncoder(embedding_size,fastapath_src, pdbPath_src, chain_src,max_len=len_input,device=device) 
trg_position_embedding_Struct = StructuralAlignedEncoder(embedding_size,fastapath_trg, pdbPath_trg, chain_trg,max_len=len_output, device=device)
# from ProteinsTransformerSideFunctions import *

#### Create Positional Encods
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
).to(device)

#whyyy 'cpu?'

wandb.init(project="Structural encoder", entity="barthelemymp")
config_dict = {
  "num_layers": 4,
  "forward_expansion": 4096,
  "batch_size": 10,
  "Encoder": "Positional"
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

# In case we would like to see just the Bleu score of the loaded model:
# score = bleu(test_data, model, protein, protein_trans, device)
# print(f"Bleu score {score * 100:.2f}")
# import sys
# sys.exit()
#sentence = "- - - - F V N R W V H Q M K T P L S V I Q L T L D E V E E P A E H I R E E L E R I R K G L E R - - - - - - - - - - - - - - - - -"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        if epoch%100==99:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="fam17Pos"+str(epoch)+".pth.tar")
            
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2]) #keep last dimension
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
            for batch_idx, batch in enumerate(test_iterator):
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
    wandb.log({"Train loss": mean_lossTrain, "Val Loss":mean_lossVal, "epoch":epoch})
wandb.finish()


#####
wandb.init(project="Structural encoder", entity="barthelemymp")
config_dict = {
  "num_layers": 4,
  "forward_expansion": 4096,
  "batch_size": 10,
  "Encoder": "Structural V2"
}
wandb.config.update(config_dict) 

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
    src_position_embedding_Struct,
    trg_position_embedding_Struct,
    device,
).to(device)



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

# In case we would like to see just the Bleu score of the loaded model:
# score = bleu(test_data, model, protein, protein_trans, device)
# print(f"Bleu score {score * 100:.2f}")
# import sys
# sys.exit()
#sentence = "- - - - F V N R W V H Q M K T P L S V I Q L T L D E V E E P A E H I R E E L E R I R K G L E R - - - - - - - - - - - - - - - - -"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    if save_model:
        if epoch%100==99:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="fam17StrV2.pth.tar")
            
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data, target= batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2]) #keep last dimension
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
    wandb.log({"Train loss": mean_lossTrain, "Val Loss":mean_lossVal, "epoch":epoch})
wandb.finish()


