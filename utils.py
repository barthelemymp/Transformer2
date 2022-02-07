import torch
from torchtext.data.metrics import bleu_score
import sys
import pandas as pd
import numpy as np
import math
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from ProteinsDataset import *
from tqdm import tqdm
def translate_sentence(model, sentence,protein,protein_trans, device, max_length=114):

    if type(sentence) == str:
        tokens = sentence.split(' ')
        #print(tokens)
    elif type(sentence) == list:
        tokens = sentence
    #print(tokens)
    #else:
        #tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, protein.init_token)
    tokens.append(protein.eos_token)

    # Go through each protein token and convert to an index
    text_to_indices = [protein.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [protein_trans.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == protein_trans.vocab.stoi["<eos>"]:
            break

    translated_sentence = [protein_trans.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, protein,protein_trans, device):
    targets = []
    outputs = []
    
    #i = 0
    for example in data:
        #print(i)
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        
        prediction = translate_sentence(model, src, protein,protein_trans,device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)
        
        #i+=1
        #print(' ')
    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
#torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch))) to save each epoch


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



def splitcsv(pathToCSV, name, repartition, shuffle=False, maxval=None):
    
    train_per = repartition[0]
    test_per = repartition[1]
    if len(repartition)>=3:
        val_per = repartition[2]
        
    path = pathToCSV + name + ".csv"
    
    df = pd.read_csv(path)
    if shuffle == True:
        df = df.sample(frac=1).reset_index(drop=True)
    
    total_size=len(df)
    
    train_size=math.floor(train_per*total_size)
    test_size=math.floor(test_per*total_size)
    if maxval:
        test_size= min(test_size, maxval)
    traintest=df.head(train_size+test_size)
    train = traintest.head(train_size)
    test= traintest.tail(test_size)
    
    train.to_csv(pathToCSV + name +'_train.csv', index = False)
    test.to_csv(pathToCSV + name +'_test.csv', index = False)
    
    if len(repartition)>=3:
        val_size=math.floor(val_per*total_size)
        if maxval:
            val_size= min(val_size, maxval)
        val=df.tail(val_size)
        val.to_csv(pathToCSV + name +'_val.csv',index = False)
    

def getLengthfromCSV(pathToFile):
    df = pd.read_csv(pathToFile)
    inputsize = len(df.iloc[1][0].split(" "))
    outputsize = len(df.iloc[1][1].split(" "))
    return inputsize, outputsize
    
def HungarianMatching(pds_val, model):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.SymbolMap["<pad>"], reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    bs = listin.shape[1]
    scoreHungarian = np.zeros((bs, bs))
    with torch.no_grad():
        for j in tqdm(range(bs)):

            inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,bs,1)
            output = model(inp_repeted, listout[:-1, :])
            output = output.reshape(-1, output.shape[2])#keep last dimension
            _, targets_Original = listout.max(dim=2)
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterionE(output, targets_Original).reshape(-1,bs).mean(dim=0)
            scoreHungarian[j,:] = loss.cpu().numpy()
    # nameH = "scoreHungarianTransformer" + str(i) + "npy"
    # np.save(nameH, scoreHungarian)
    # scoH = scipy.optimize.linear_sum_assignment(scoreHungarian)
    # scoreMatching = sum(scoH[0]==scoH[1])
    return scoreHungarian

def makebatchList(tot, bs):
    batchIndex=[]
    nbatch = tot//bs
    last = tot%bs
    for i in range(nbatch):
        start = i*bs
        end =start+bs
        batchIndex.append(range(start, end))
    if last!=0:
        start = nbatch*bs
        end = start+last
        batchIndex.append(range(start, end))
    return batchIndex
        
    

def HungarianMatchingBS(pds_val, model, batchs):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.SymbolMap["<pad>"], reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    tot = listin.shape[1]
    scoreHungarian = np.zeros((tot, tot))
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        for j in tqdm(range(tot)):
            for batch in batchIndex:
                if model.onehot:
                    _, targets_Original = listout[:,batch].max(dim=2)
                    inp_repeted = listin[:,j,:].unsqueeze(1).repeat(1,len(batch),1)
                else:
                    targets_Original= listout[:,batch]
                    inp_repeted = listin[:,j].unsqueeze(1).repeat(1,len(batch))
                
                output = model(inp_repeted, listout[:-1, batch])
                output = output.reshape(-1, output.shape[2])#keep last dimension

                targets_Original = targets_Original[1:].reshape(-1)
                loss = criterionE(output, targets_Original).reshape(-1,len(batch)).mean(dim=0)
                scoreHungarian[j,batch] = loss.cpu().numpy()
    # nameH = "scoreHungarianTransformer" + str(i) + "npy"
    # np.save(nameH, scoreHungarian)
    # scoH = scipy.optimize.linear_sum_assignment(scoreHungarian)
    # scoreMatching = sum(scoH[0]==scoH[1])
    return scoreHungarian


def distanceTrainVal(pds_val, pds_train):
    #.t() because of batch first
    if pds_val.onehot:
        proteinIN1 = pds_train[:][0].max(dim=2)[1].float().t()
        proteinOUT1 = pds_train[:][1].max(dim=2)[1].float().t()
        proteinIN2 = pds_val[:][0].max(dim=2)[1].float().t()
        proteinOUT2 = pds_val[:][1].max(dim=2)[1].float().t()
    else:
        proteinIN1 = pds_train[:][0].float().t()
        proteinOUT1 = pds_train[:][1].float().t()
        proteinIN2 = pds_val[:][0].float().t()
        proteinOUT2 = pds_val[:][1].float().t()
    DistanceIN = torch.cdist(proteinIN2, proteinIN1, p=0.0)
    DistanceOUT = torch.cdist(proteinOUT2, proteinOUT1, p=0.0)
    return DistanceIN, DistanceOUT
    
def accuracy(batch, output, onehot=False):
    bs = output.shape[1]
    ra = range(bs)
    if onehot==False:
        proteinOUT1 = batch[1][1:-1,:]
        proteinOUT1 = proteinOUT1.float().t()
        
        proteinOUT2 = output.max(dim=2)[1][:-1,:]
        proteinOUT2 = proteinOUT2.float().t()

    Distance = torch.cdist(proteinOUT1, proteinOUT2, p=0.0)[ra,ra]
    return torch.sum(Distance)
    
    

        
def ConditionalEntropyEstimator(pds_val, model, batchs=100, returnAcc=False):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds_val.SymbolMap["<pad>"], reduction='none')
    data = getPreciseBatch(pds_val, torch.tensor(range(len(pds_val))))
    listin, listout = data[0], data[1]
    tot = listin.shape[1]
    max_len = listout.shape[0]
    batchIndex = makebatchList(tot, batchs)
    with torch.no_grad():
        entropylist =[]
        acc = 0
        for batch in tqdm(batchIndex):
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
                acc+=accuracy(pds_val[batch], sampled[1:]).item()
                output = model(listin[:,batch], sampled[:-1, :])[:-1]
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]#listout[:,batch]
                targets_Original = targets_Original[1:-1].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).sum(dim=0)
                entropylist+=[Entropy[i] for i in range(len(Entropy))]
        print(acc/len(pds_val))
        meanEntropy = sum(entropylist)/len(entropylist)
    if returnAcc:
        return meanEntropy, acc/len(pds_val)
    else:
        return meanEntropy
    
    
   
def ConditionalEntropyEstimatorGivenInp(inp, model, pad, max_len,nseq=1000, batchs=1000, returnAcc=False):
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
        for batch in tqdm(batchIndex):
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
                #sampled = sampled.max(dim=2)[1]
                # acc+=accuracy(pds_val[batch], sampled[1:]).item()
                output = model(listin[:,batch], sampled[:-1, :])[:-1]
                output = output.reshape(-1, output.shape[2])
                targets_Original = sampled.max(dim=2)[1]#listout[:,batch]
                targets_Original = targets_Original[1:-1].reshape(-1)
                Entropy = criterionE(output, targets_Original).reshape(-1,len(batch)).sum(dim=0)
                entropylist += [Entropy[i] for i in range(len(Entropy))]
        # print(acc/nseq)
        meanEntropy = sum(entropylist)/len(entropylist)
    if returnAcc:
        return meanEntropy, acc/nseq
    else:
        return meanEntropy
    

        
        




# if self.auto_collation:
#     data = []
#     for _ in possibly_batched_index:
#         try:
#             data.append(next(self.dataset_iter))
#         except StopIteration:
#             self.ended = True
#             break
#     if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
#         raise StopIteration
# else:
#     data = next(self.dataset_iter)
# return self.collate_fn(data)