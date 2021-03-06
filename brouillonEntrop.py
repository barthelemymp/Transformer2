# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:36:03 2022

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
torch.set_num_threads(6)
print("import done")
#torch.functional.one_hot
pathtoFolder = "testFakedata.csv"
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 5
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.10
forward_expansion = 112
src_vocab_size = 8#len(protein.vocab) 
trg_vocab_size = 8#len(protein_trans.vocab) 
embedding_size = 10#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token

repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =50
Unalign = False

wd_list = [0.0]#, 0.00005]
# ilist = [46, 69, 71,157,160,251, 258, 17]
onehot=False
wd=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




##### Training simple 
pathTofile = pathtoFolder
inputsize, outputsize = getLengthfromCSV(pathTofile)
os.path.isfile(pathTofile)
count +=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Dataset
train_path = pathtoFolder

#add 2 for start and end token 
len_input = inputsize + 2
len_output =outputsize + 2
pds_train = ProteinTranslationDataset(train_path, mapstring = "ABCDE",device=device, Unalign=Unalign,filteringOption='or', returnIndex=True,onehot=onehot)
ntrain = len(pds_train)

# ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
# print("score", i)
# print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

train_iterator = DataLoader(pds_train, batch_size=batch_size,
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

alpha = -0.0

step = 0
step_ev = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 5e-5

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)
        


ntest = 0
inp, target = pds_train[0][0].unsqueeze(1), pds_train[0][1].unsqueeze(1)
targets = target.repeat(1,8*8*8*8)
count=0
for a1 in range(8):
    for a2 in range(8):
        for a3 in range(8):
            for a4 in range(8):
                targets[1,count] = a1
                targets[2,count] = a2
                targets[3,count] = a3
                targets[4,count] = a4
                count+=1
                

print(targets)
inps = inp.repeat(1,8*8*8*8)



def exactEntropy(inps, targets, model):
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
    model.eval()
    with torch.no_grad():
        output = model(inps, targets[:-1, :])[:-1]
        output = output.reshape(-1, output.shape[2])#keep last dimension
        targets_Original= targets
        targets_Original = targets_Original[1:-1].reshape(-1)
        loss =criterionE(output, targets_Original).reshape(-1,targets.shape[1])
    print(loss.shape)
    loss = loss.sum(dim=0)
    probseq = torch.exp(-1*loss).view(1,-1,1)
    entropy = (loss)*torch.exp(-1*loss)
    entropy = torch.sum(entropy)
    print(torch.sum(torch.exp(-1*loss)))
    return entropy

exactEntropy(inps, targets, model)

        
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])

for epoch in range(500+1):
    _=model.train()
    lossesCE = []
    accuracyTrain = 0
    for batch_idx, batch in enumerate(train_iterator):
        optimizer.zero_grad()
        lossCE, lossEntropy, acc = ConditioalEntropyMatchingLoss(batch, model, criterion, device, samplingMultiple=1)
        accuracyTrain += acc
        lossesCE.append(lossCE.item())
        loss = lossCE + alpha * lossEntropy
        loss.backward()
        _=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()
    if epoch%100==0:
        _=model.eval()
        entropytest = ConditionalEntropyEstimatorGivenInp(pds_train[0][0], model, pds_train.SymbolMap["<pad>"], targets.shape[0],nseq=50000, batchs=300, returnAcc=False)
        print(f"[Epoch {epoch} / {num_epochs}]", entropytest, exactEntropy(inps, targets, model), sum(lossesCE)/len(lossesCE))


    
#test on first
def toString(li):
    s = ""
    for i in li:
        s+=str(i.item())
    return s



maps = { toString(targets[1:-1,i]):i for i in range(targets.shape[1])}


freqsampled = np.zeros(len(maps))
count=0
for j in tqdm.tqdm(range(500)):
    sampledT = model.sample(inps, targets.shape[0], nsample=1, method="simple").max(dim=2)[1]
    for i in range(sampled.shape[1]):
        freqsampled[maps[toString(sampledT[1:-1,i])]]+=1
        count+=1
        


freqsampled/=count



criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
model.eval()
with torch.no_grad():
    output = model(inps, targets[:-1])[:-1]
    output = output.reshape(-1, output.shape[2])#keep last dimension
    targets_Original= targets
    targets_Original = targets_Original[1:-1].reshape(-1)
    loss =criterionE(output, targets_Original).reshape(-1,targets.shape[1])

print(loss.shape)
loss = loss.sum(dim=0)
probseq = torch.exp(-1*loss)

freqcalc = np.zeros(len(maps))

for i in range(sampled.shape[1]):
    freqcalc[maps[toString(targets[1:-1,i])]]=probseq[i]

        
        
        
sampled = model.sample(inps, targets.shape[0], nsample=1, method="simple")
freq = sampled.mean(dim=1)
sampled = model.sample(inps, targets.shape[0], nsample=1, method="simple")
freq += sampled.mean(dim=1)
sampled = model.sample(inps, targets.shape[0], nsample=1, method="simple")
freq += sampled.mean(dim=1)
freq/=3
probseq = torch.exp(-1*loss).view(1,-1,1)
print(torch.sum(probseq))
weighted = torch.nn.functional.one_hot(targets, num_classes=sampled.shape[2])
weighted = weighted * probseq
weighted = weighted.sum(dim=1)

print(weighted)
print(freq)




def ConditionalEntropyEstimatorGivenInp(inp, model, pad, max_len,nseq=1000, batchs=100, returnAcc=False):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pad, reduction='none')
    # data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    # listin, listout = data[0], data[1]
    listin = inp.unsqueeze(1).repeat(1,nseq)
    tot = listin.shape[1]
    # max_len = listout.shape[0]
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        entropylist =[]
        acc = 0
        for batch in tqdm.tqdm(batchIndex):
            if model.onehot:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                output = model(listin[:,batch], sampled[:-1, :])
                output = output.reshape(-1, output.shape[2])
                #_, targets_Original = listout[:,batch].max(dim=2)
                targets_Original = sampled.max(dim=2)[1]
                targets_Original = targets_Original[1:].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).mean()
                entropylist.append(Entropy)
                # inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,len(batch),1)
            else:
                sampled = model.sample(listin[:,batch], max_len, nsample=1, method="simple")
                # acc+=accuracy(pds_val[batch], sampled[1:]).item()
                output = model(listin[:,batch], sampled[:-1, :])[:-1]
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]#listout[:,batch]
                targets_Original = targets_Original[1:-1].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).sum(dim=0)
                entropylist+=[Entropy[i] for i in range(len(Entropy))]
        # print(acc/nseq)
        meanEntropy = sum(entropylist)/len(entropylist)
    if returnAcc:
        return meanEntropy, acc/nseq
    else:
        return meanEntropy




entropy = (loss)*torch.exp(-1*loss)
entropy = torch.sum(entropy)
print(entropy)
entropytest = ConditionalEntropyEstimatorGivenInp(pds_train[0][0], model, pds_train.SymbolMap["<pad>"], targets.shape[0],nseq=100000, batchs=300, returnAcc=False)
print(entropytest)
ConditionalEntropyEstimatorGivenInp(pds_train[0][0], model, pds_train.SymbolMap["<pad>"], targets.shape[0],nseq=100000, batchs=300, returnAcc=False)
exactEntropy(inps, targets, model)




##### Sample

sos = torch.nn.functional.one_hot(inp[0,0], num_classes=model.trg_vocab_size)
eos = torch.nn.functional.one_hot(inp[-1,0], num_classes=model.trg_vocab_size)
inp_repeted = inps
nsample = inps.shape[1]
outputs = torch.zeros(max_len, nsample, model.trg_vocab_size).to(model.device)
outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
for i in range(1,max_len):
    print(i)
    print(outputs[:, 0])
    output = model.forward(inp_repeted, outputs[:i])
    print(output[:,0,:])
    # prob = torch.nn.functional.softmax(output.clone().detach(),dim=2).reshape(-1,model.trg_vocab_size)
    prob = output.clone().detach().reshape(-1,model.trg_vocab_size)
    best_guess = torch.distributions.Categorical(logits=prob)#(prob, nsample, replacement=True)
    best_guess = torch.nn.functional.one_hot(best_guess, num_classes=model.trg_vocab_size).reshape(-1,nsample,model.trg_vocab_size)
    outputs[i,:,:]= best_guess[-1,:,:]

outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)

    return outputs     
        