
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
import matplotlib.pyplot as plt
import pandas as pd
print("import done")


def evaluateCE_matrix(pds_val, model):
    criterion = nn.CrossEntropyLoss(ignore_index=pds_val.padIndex, reduction ='none')
    accuracyTrain = np.zeros(len(pds_val))
    batchlist = makebatchList(len(pds_val), 100)
    score = torch.zeros((pds_val.outputsize-1, len(pds_val))).to(model.device)
    with torch.no_grad():
        for batch in batchlist:
            inp_data, target, _= pds_val[batch]
            output = model(inp_data, target[:-1, :])
            accuracyTrain[batch] += accuracy(pds_val[batch], output, onehot=False).item()
            output = output.reshape(-1, output.shape[2])#keep last dimension
            targets_Original= target
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterion(output, targets_Original).reshape(-1,len(batch))
            print(loss.shape)
            score[ :, batch] = loss
    return score

pdbtracker = pd.read_csv("pdbtracker.csv")
#torch.functional.one_hot
pathtoFolder = "/Data/DomainsInter/processed/"
torch.set_num_threads(12)
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 1
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 21#len(protein.vocab) 
trg_vocab_size = 21#len(protein_trans.vocab) 
embedding_size = 55#len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token

repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.01, 0.1]
wd_list = [0.0]#, 0.00005]
ilist = [46, 69, 71,157,160,251, 258, 17]
onehot=False
wd=0.0

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")    
#             ilist.append(i)
            
i = 304
save_model = False
onehot=False



modelpath = "models/LargeL2/Large_fam"+str(i)+"_wd0_epoch"+str(4000)+".pth.tar"

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

num_heads = 1
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 21#len(protein.vocab) 
trg_vocab_size = 21#len(protein_trans.vocab) 
embedding_size = 105



pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
inputsize, outputsize = getLengthfromCSV(pathTofile)
os.path.isfile(pathTofile)
print("ddi", i, " is running")
name = "combined_MSA_ddi_" +str(i)+"_joined"

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

train_iterator = DataLoader(pds_train, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
test_iterator = DataLoader(pds_test, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)
val_iterator = DataLoader(pds_val, batch_size=batch_size,
                shuffle=True, num_workers=0, collate_fn=default_collate)


# Model hyperparameters

src_pad_idx = pds_train.padIndex#pds_train.SymbolMap["<pad>"]#"<pad>"# protein.vocab.stoi["<pad>"] 
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

step = 0
step_ev = 0
device = torch.device("cpu")#("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 5e-5

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)
        
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
#pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pds_train.padIndex)

load_checkpoint(torch.load(modelpath, map_location=torch.device('cpu')), model, optimizer)


pdssample = sampleDataset(model, pds_train, len_output, multiplicative =1)

CEtrain = evaluateCE_matrix(pds_train, model).sum(dim=0).cpu().numpy()
#CEval = np.exp(-1*evaluateCE_matrix(pds_val, model).sum(dim=0).cpu().numpy())
CEsample = evaluateCE_matrix(pdssample, model).sum(dim=0).cpu().numpy()
order = np.argsort(CEtrain)
x = np.array(range(len(order)))
y = CEtrain[order]
y2 = CEsample[order]

plt.xlabel("Training Points", fontsize=18)
plt.ylabel("Probability", fontsize=18)
plt.title("Landscape" , fontsize=18)
plt.scatter(x,y, alpha=0.8, color="blue", label="Training Points")
plt.scatter(x,y2, alpha=0.8, color="red", label="Sampled Points")
#plt.scatter(x,y3, alpha=0.3, color="green", label="Reyni")
plt.tick_params(axis='both', labelsize=18)
plt.legend(fontsize=18)
plt.savefig("energy"+str(i)+".pdf")
