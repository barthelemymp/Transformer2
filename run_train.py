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
torch.set_num_threads(4)

def get_params(params):

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--fam', type=int, help="msa to use")
    parser.add_argument('--nlayer', type=int, default=2,help="Path to expression matrix")
    parser.add_argument('--embedding_size', type=int, default=55, help="Method for features selection")
    parser.add_argument('--nhead', type=int, default=5, help="Method for features selection")
    parser.add_argument('--batch_size', type=int, default=32, help="Method for features selection")
    parser.add_argument('--forward_expansion', type=int, default=2048, help="Number of subsets")
    parser.add_argument('--num_epochs', type=int, default=5000, help="Number of features when applied to subset")
    parser.add_argument('--dir_save', type=str, default="", help="Method for features selection")
    parser.add_argument('--dir_load', type=str, default="", help="MI info save path")
    args = parser.parse_args(params)

    return args

def main(params):

    
    
    # Params
    opts = get_params(params)
    i = opts.fam
    nlayer = opts.nlayer
    num_encoder_layers = nlayer
    num_decoder_layers = nlayer
    embedding_size = opts.embedding_size
    if embedding_size>0:
        onehot=False
    Unalign = False
    num_heads = opts.nhead
    batch_size = opts.batch_size
    forward_expansion = opts.forward_expansion
    num_epochs= opts.num_epochs
    src_vocab_size = 25
    trg_vocab_size = 25
    dropout = 0.10
    load_model= bool(opts.dir_save)
    save_model= bool(opts.dir_load)
    
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
    # ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH = ARDCA(pds_train, pds_test, pds_val)
    # print("score", i)
    # print(i, ardcaTrain, ardcaTest, ardcaVal, acctrain, acctest, accval, ardcascoreH)

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
    wandb.init(project="Transformer Simple large Fam", entity="barthelemymp")
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
            
            if save_model:
                save_checkpoint(checkpoint, filename=opts.dir_save)
        
    
    wandb.finish()
    

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])