
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
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.10
forward_expansion = 4096
repartition = [0.7, 0.15, 0.15]
#EPOCHS 
num_epochs = 2500
Unalign = False
alphalist=[0.0, 0.001, 0.01, 0.1, 1.0]
i=46


for alpha in alphalist:
    pathTofile = pathtoFolder+ "combined_MSA_ddi_" +str(i)+"_joined.csv"
    if os.path.isfile(pathTofile)==False:
        continue
    inputsize, outputsize = getLengthfromCSV(pathTofile)
    os.path.isfile(pathTofile)
    if abs(outputsize - inputsize)< 0.2*inputsize:
        count +=1
        print("ddi", i, " is running")
        name = "combined_MSA_ddi_" +str(i)+"_joined"
        #        splitcsv(pathtoFolder, name, repartition, shuffle=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        #Dataset
        train_path = pathtoFolder + name +'_train.csv'
        val_path = pathtoFolder + name +'_val.csv'
        test_path = pathtoFolder + name +'_test.csv'
        
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
        if src_vocab_size!=trg_vocab_size:
            print("the input vocabulary differs from output voc", src_vocab_size, trg_vocab_size)
            continue
        assert src_vocab_size==trg_vocab_size, "the input vocabulary differs from output voc"
        assert src_vocab_size<=25, "Something wrong with length of vocab in input s."
        assert trg_vocab_size<=25, "Something wrong with length of vocab in output s."
        
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
        wandb.init(project="alpha Bipartite", entity="barthelemymp")
        config_dict = {
          "num_layers": 6,
          "forward_expansion": 4096,
          "batch_size": batch_size,
          "Encoder": "Positional",
          "Family":i,
          "len input":len_input,
          "len output":len_output,
          "scheduler": "30 on train",
          "loss": "CE+Matching BCE",
          "alphaParameter":alpha,
          "weightdecay":0.0
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
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=30, verbose=True
        # )
        
        # The following line defines the pad token that is not taken into 
        # consideration in the computation of the loss - cross entropy
        pad_idx = "<pad>"#protein.vocab.stoi["<pad>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
        for epoch in range(num_epochs+1):
            print(f"[Epoch {epoch} / {num_epochs}]")
            model.train()
            lossesCE = []
            lossesMatching = []

            for batch_idx, batch in enumerate(train_iterator):
                optimizer.zero_grad()
                loss, lossMatching = makeBicontrastiveCELoss(batch, model, criterion, alpha=0.0)
                lossesCE.append(loss.item())
                lossesMatching.append(lossMatching.item())
                Totloss = loss + alpha*lossMatching 
                Totloss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
                optimizer.step()
            mean_lossCETrain = sum(lossesCE) / len(lossesCE)
            mean_lossMatchingTrain = sum(lossesMatching) / len(lossesMatching)
            step += 1
            # scheduler.step(mean_lossCETrain)
            model.eval()
            lossesCE_eval = []
            lossesMatching_eval = []
            if epoch%1==0:
                with  torch.no_grad():
                    for batch_idx, batch in enumerate(val_iterator):
                        lossCE, lossMatching =  makeBicontrastiveCELoss(batch, model, criterion, alpha=0.0)
                        lossesCE_eval.append(lossCE.item())
                        lossesMatching_eval.append(lossMatching.item())
                    mean_lossCEVal = sum(lossesCE_eval) / len(lossesCE_eval)
                    mean_lossMatchingVal = sum(lossesMatching_eval) / len(lossesMatching_eval)
                    step_ev +=1
            wandb.log({"Train loss CE": mean_lossCETrain,  "Val loss CE": mean_lossCEVal,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha, "Train loss Matching": mean_lossMatchingTrain, "epoch":epoch})#,"Val Loss Matching":mean_lossMatchingVal, "alpha":alpha "Train loss Matching": mean_lossMatchingTrain,
            if epoch%100==0:
                criterionE = nn.CrossEntropyLoss(ignore_index=pds_train.SymbolMap["<pad>"], reduction='none')
                listin, listout = pds_val[:]
                bs = listin.shape[1]

                scoreHungarian = np.zeros((bs, bs))
                with torch.no_grad():
                    for j in range(bs):
                        print(j)
                        inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,bs,1)
                        output = model(inp_repeted, listout[:-1, :])
                        output = output.reshape(-1, output.shape[2])#keep last dimension
                        _, targets_Original = listout.max(dim=2)
                        targets_Original = targets_Original[1:].reshape(-1)
                        loss = criterionE(output, targets_Original).reshape(-1,bs).mean(dim=0)
                        scoreHungarian[j,:] = loss.cpu().numpy()
                # nameH = "scoreHungarianTransformer" + str(i) + "npy"
                # np.save(nameH, scoreHungarian)
                scoH = scipy.optimize.linear_sum_assignment(scoreHungarian)
                scoreMatching = sum(scoH[0]==scoH[1])
                scoHexp = scipy.optimize.linear_sum_assignment(np.exp(scoreHungarian))
                scoreMatchingexp = sum(scoHexp[0]==scoHexp[1])
                wandb.log({"scoreMatching Val": scoreMatching, "scoreMatching exp Val": scoreMatchingexp, "epoch":epoch})

                listin, listout = pds_train[:]
                bs = listin.shape[1]
                scoreHungarian = np.zeros((bs, bs))
                with torch.no_grad():
                    for j in range(bs):
                        print(j)
                        inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,bs,1)
                        output = model(inp_repeted, listout[:-1, :])
                        output = output.reshape(-1, output.shape[2])#keep last dimension
                        _, targets_Original = listout.max(dim=2)
                        targets_Original = targets_Original[1:].reshape(-1)
                        loss = criterionE(output, targets_Original).reshape(-1,bs).mean(dim=0)
                        scoreHungarian[j,:] = loss.cpu().numpy()
                # nameH = "scoreHungarianTransformer" + str(i) + "npy"
                # np.save(nameH, scoreHungarian)
                scoH = scipy.optimize.linear_sum_assignment(scoreHungarian)
                scoreMatching = sum(scoH[0]==scoH[1])
                scoreMatchingexp = sum(scoHexp[0]==scoHexp[1])
                # wandb.log({"scoreMatching Val": scoreMatching, "scoreMatching exp Val": scoreMatchingexp, "epoch":epoch})
                wandb.log({"scoreMatchingTrain": scoreMatching, "epoch":epoch})
        wandb.finish()
               
        
        
        
