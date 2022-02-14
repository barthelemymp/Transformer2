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
pathtoFolder = "/home/feinauer/Datasets/DomainsInter/processed/"
torch.set_num_threads(4)
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
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

repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs =5000
Unalign = False
alphalist=[0.0, 0.01, 0.1]
wd_list = [0.0]#, 0.00005]
ilist = [46, 69, 71,157,160,251, 258, 17]
onehot=False
wd=0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#             ilist.append(i)
            
ilist = [46, 69, 71,157,160,251, 258, 17] 
save_model = True
onehot=False
for i in range(1500):
    modelpath = "TransSimple_fam"+str(i)+".pth.tar"
    if os.path.isfile(modelpath):
        
        
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
        wandb.init(project="Transformer Simple Matching resilience", entity="barthelemymp")
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
          "loss": "CE"
        }
        wandb.config.update(config_dict) 
        
        step = 0
        step_ev = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        learning_rate = 5e-5
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"])

        load_checkpoint(torch.load(modelpath), model, optimizer)
        
        criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        model.eval()
        criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
        scoreHungarianVal_Ardca = ARDCA_returnmatrix(pds_train, pds_test, pds_val)
        print("score", i)
        print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

        for siz in range(10, len(pds_val), 10):
            subscoreH = scoreHungarianVal[:siz, :siz]
            scoHVal = scipy.optimize.linear_sum_assignment(subscoreH)
            scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
            
            subscoreH = scoreHungarianVal_Ardca[:siz, :siz]
            scoHVal = scipy.optimize.linear_sum_assignment(subscoreH)
            scoreMatchingVal_Ardca = sum(scoHVal[0]==scoHVal[1])
            
        wandb.log({"siz": siz, "scoreMatchingVal Trans":scoreMatchingVal, "scoreMatchingVal Ardca":scoreMatchingVal_Ardca})
        
        # In c