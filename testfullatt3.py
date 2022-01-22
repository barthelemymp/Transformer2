# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 02:05:55 2021

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
from utils import *
print("import done")
#torch.functional.one_hot
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
num_epochs = 1400
Unalign = False
for i in range(0,300):
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
        

        #If you want to save the model, u can use this additional parameter 
        #to save those better than 
        #min_mean_loss = 1.2
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        #Dataset
        train_path = pathtoFolder + name +'_train.csv'
        val_path = pathtoFolder + name +'_val.csv'
        test_path = pathtoFolder + name +'_test.csv'

# =============================================================================
#         #number of amino acids in the input sequence
#         len_input = 101 
#         len_output = 118  
# =============================================================================
        
        #add 2 for start and end token 
        len_input = inputsize + 2
        len_output =outputsize + 2

        pds_train = ProteinTranslationDataset(train_path, device=device, Unalign=Unalign)
        pds_test = ProteinTranslationDataset(test_path, device=device, Unalign=Unalign)
        pds_val = ProteinTranslationDataset(val_path, device=device, Unalign=Unalign)
        
        train_iterator = DataLoader(pds_train, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        test_iterator = DataLoader(pds_test, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)
        val_iterator = DataLoader(pds_val, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=default_collate)

# =============================================================================
#         protein = Field(init_token="<sos>", eos_token="<eos>", postprocessing=x:torch.functional.one_hot(x,25))
#         protein_trans = Field(init_token="<sos>", eos_token="<eos>", postprocessing=x:torch.functional.one_hot(x,25))
# 
#         data_fields = [('src', protein), ('trg', protein_trans)]
#         train,val,test = TabularDataset.splits(path='./', train=train_path , validation=val_path , test=test_path ,format='csv', fields=data_fields)
#         protein.build_vocab(train, min_freq=1)
#         protein_trans.build_vocab(train,  min_freq=1)
# =============================================================================
        

        
# =============================================================================
#         train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#             (train, val, test),
#             batch_size=batch_size,
#             device=device,
#             sort = False,
#             sort_within_batch=True,
#             sort_key=lambda x: len(x.src),
#             
#         )
# =============================================================================
# =============================================================================
#         for batch_idx, batch in enumerate(train_iterator):
# #            print(batch.src.device)
#             batch.src = batch.src.to(device, non_blocking=True)
# #            print(batch.src.device)
#             batch.trg = batch.trg.to(device, non_blocking=True)
#         for batch_idx, batch in enumerate(test_iterator):
#             batch.src = batch.src.to(device, non_blocking=True)
#             batch.trg = batch.trg.to(device, non_blocking=True)
#         for batch_idx, batch in enumerate(valid_iterator):
#             batch.src = batch.src.to(device, non_blocking=True)
#             batch.trg = batch.trg.to(device, non_blocking=True)
# =============================================================================
        
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
        
        model = Full_att_Transformer(
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
            InterTransformersBackprop=True,
            sampleInter="gumbel"
        ).to(device)


        #whyyy 'cpu?'
        wandb.init(project="Test full attention 2", entity="barthelemymp")
        config_dict = {
          "num_layers": 6,
          "forward_expansion": 4096,
          "batch_size": batch_size,
          "Encoder": "Positional",
          "Family":i,
          "len input":len_input,
          "len output":len_output,
          "output full att": "gumbel",
          "shared encoder":"False",
          "backprop through ":"True"
          
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.3, patience=50, verbose=True
        # )
        
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
                if epoch==1399:
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, filename="fam"+str(i)+"Pos_gumbel.pth.tar")
        
                #step = 0
                #model.eval()
                #translated_sentence = translate_sentence(
                     #model, sentence, protein, protein_trans, device, max_length=114
                #)
        
            #print(f"Translated example sentence: \n {translated_sentence}")
            model.train()
            losses = []
            losses2 = []

        
            for batch_idx, batch in enumerate(train_iterator):
# =============================================================================
#                 inp_data = batch.src#.to(device)
#                 target = batch.trg#.to(device)
# =============================================================================
#                print(batch.src.device)
                inp_data, target= batch[0], batch[1]
        
                output,output2 = model(inp_data, target[:-1, :])

                output = output.reshape(-1, output.shape[2])#keep last dimension
                output2 = output2.reshape(-1, output2.shape[2])
                _, targets_Original = target.max(dim=2)
                targets_Original = targets_Original[1:].reshape(-1)

        
                optimizer.zero_grad()
        
                loss = criterion(output, targets_Original)
                loss2 = criterion(output2, targets_Original)
                losses.append(loss.item())
                losses2.append(loss2.item())
                lossTot = loss + loss2
                lossTot.backward()
        
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        
                # Gradient descent step
                optimizer.step()
                
        
                # plot to tensorboard
        
            mean_lossTrain = sum(losses) / len(losses)
            mean_lossTrain2 = sum(losses2) / len(losses2)
            # writer.add_scalar("Training loss", mean_loss, global_step=step)
            step += 1
        
            # scheduler.step(mean_lossTrain)
        
            model.eval()
            #translated_sentence = translate_sentence(
                 #model, sentence, protein, protein_trans, device, max_length=114
            #)
        
            #print(f"Translated example sentence: \n {translated_sentence}")
            losses_eval = []
            losses_eval2 = []
            losses_eval3 = []
            if epoch%1==0:
                with  torch.no_grad():
                    for batch_idx, batch in enumerate(val_iterator):
                        inp_data, target= batch[0], batch[1]
                        print(inp_data.device)
# =============================================================================
                        inp_data = inp_data.to(device)
#                         target = batch.trg.to(device)
# =============================================================================
                        print(inp_data.device)
                
                        output,output2 = model(inp_data, target[:-1, :])
                        output3 = model.full_trans_forward(inp_data, output2)
                        output = output.reshape(-1, output.shape[2]) #keep last dimension
                        output2 = output2.reshape(-1, output2.shape[2])
                        output3 = output3.reshape(-1, output3.shape[2])
                        _, targets_Original = target.max(dim=2)
                        targets_Original = targets_Original[1:].reshape(-1)
                        
                
                        loss_eval = criterion(output, targets_Original)
                        loss_eval2 = criterion(output2, targets_Original)
                        loss_eval3 = criterion(output3, targets_Original)
                        losses_eval.append(loss_eval.item())
                        losses_eval2.append(loss_eval2.item())
                        losses_eval3.append(loss_eval3.item())
                        
                    mean_lossVal = sum(losses_eval) / len(losses_eval)
                    mean_lossVal2 = sum(losses_eval2) / len(losses_eval2)
                    mean_lossVal3 = sum(losses_eval3) / len(losses_eval3)
                    step_ev +=1
            wandb.log({"Train loss": mean_lossTrain, "Val Loss":mean_lossVal, "Train loss 2": mean_lossTrain2, "Val Loss2":mean_lossVal2, "Val Loss3":mean_lossVal3, "epoch":epoch})
        
            #To save model configuration for which the loss is the lowest on the validation set:
            #if mean_loss < min_mean_loss:
                #min_mean_loss = mean_loss
                
                #checkpoint = {
                    #"state_dict": model.state_dict(),
                    #"optimizer": optimizer.state_dict(),
                #}
                #save_checkpoint(checkpoint)
        wandb.finish()
        
