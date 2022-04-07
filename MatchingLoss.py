# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:02:01 2021

@author: bartm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd
from ProteinsDataset import *
import scipy.optimize
from utils import *
def makesquaredLoss(batch, model, criterion, criterionMatching, device, accumulate=False, alpha=0.0):
    inp_data, target= batch[0], batch[1]
    bs = inp_data.shape[1]
    lossMatrix = torch.zeros((bs,bs)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor(range(bs)).to(device)
    for i in range(bs):
        print("makesquared", i)
        inp_repeted = inp_data[:,i,:].unsqueeze(1).repeat(1,bs,1)
        # print(i, inp_repeted.shape)
        output = model(inp_repeted, target[:-1, :])
        output = output.reshape(-1, output.shape[2])#keep last dimension
        _, targets_Original = target.max(dim=2)
        targets_Original = targets_Original[1:].reshape(-1)
        loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
        lossMatrix[i,:] = loss
        LossCE += loss[i]
        if accumulate:
            lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
            Totloss = (loss[i] + alpha*lossMatchingtemp)/bs
            Totloss.backward()
            lossMatching += lossMatchingtemp.clone().detach()/bs
    if accumulate==False:        
        LossCE /=bs
        lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
        lossMatching = criterionMatching(lossMatrix, targetMatching)# + 0.5*criterionMatching(torch.t(lossMatrix), targetMatching)
        return LossCE, lossMatching
    else:
        LossCE /=bs
        return LossCE.clone().detach(), lossMatching
    
    
        
def makeBicontrastiveLoss(batch, model, criterion, alpha=0.0):
    inp_data, target= batch[0], batch[1]
    bs = inp_data.shape[1]
    perm = torch.tensor(range(1,bs+1))
    perm[-1] = 0
    targetFake = target[:,perm,:]
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])#keep last dimension
    _, targets_Original = target.max(dim=2)
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original)
    outputFake = model(inp_data, targetFake[:-1, :])
    outputFake = outputFake.reshape(-1, outputFake.shape[2])#keep last dimension
    _, targets_OriginalFake = targetFake.max(dim=2)
    targets_OriginalFake = targets_OriginalFake[1:].reshape(-1)
    lossFake = -1*criterion(outputFake, targets_OriginalFake)

    return loss, lossFake
    
        
def makeBicontrastiveCELoss(batch, model, criterion, alpha=0.0):
    
    BCE = nn.CrossEntropyLoss()
    inp_data, target= batch[0], batch[1]
    bs = inp_data.shape[1]
    targetMatching = torch.tensor(range(bs)).to(inp_data.device)
    targetMatching = torch.zeros_like(targetMatching)
    perm = torch.tensor(range(1,bs+1))
    perm[-1] = 0
    targetFake = target[:,perm,:]
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])#keep last dimension
    _, targets_Original = target.max(dim=2)
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
    outputFake = model(inp_data, targetFake[:-1, :])
    outputFake = outputFake.reshape(-1, outputFake.shape[2])#keep last dimension
    _, targets_OriginalFake = targetFake.max(dim=2)
    targets_OriginalFake = targets_OriginalFake[1:].reshape(-1)
    lossFake = criterion(outputFake, targets_OriginalFake).reshape(-1,bs).mean(dim=0)
    lossCE = torch.mean(loss)
    Bipred = torch.cat([-1*loss.unsqueeze(1), -1*lossFake.unsqueeze(1)],dim=1)
    lossMatching = BCE(Bipred, targetMatching)

    return lossCE, lossMatching

    

def sampleHardNegative(losses, ind, numberContrastive, beta):
    """ to do """
    return


def MinHardNegative(losses, ind, numberContrastive):
    losses = torch.tensor(losses)
    out = torch.sort(losses)[1]
    #print("is ind in the less loss", ind in out[:numberContrastive])
    out = out[out!=ind]
    out = out[:numberContrastive]
    out[numberContrastive-1] = ind
    return out




def checklossContrastive(model, pds, indexIN, indexOUT):
    model.eval()
    criterionE = nn.CrossEntropyLoss(ignore_index=pds.SymbolMap["<pad>"], reduction='none')
    inputProt = pds[indexIN][0].unsqueeze(1)
    outputProt = pds[indexOUT][1].unsqueeze(1)
    with torch.no_grad():
            output = model(inputProt, outputProt[:-1, :])
            output = output.reshape(-1, output.shape[2])#keep last dimension
            _, targets_Original = outputProt.max(dim=2)
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterionE(output, targets_Original).reshape(-1,1).mean(dim=0)
    return loss
    

def HardNegativeSamplingContrastiveMatchingLoss(batch,
                                                model,
                                                scoreHungarian,
                                                pds,
                                                criterion,
                                                criterionMatching,
                                                device,
                                                accumulate=False,
                                                alpha=0.0,
                                                numberContrastive="batch_size",
                                                negativesampler=MinHardNegative,
                                                parcimony=False):
    
    scoH = scipy.optimize.linear_sum_assignment(scoreHungarian)
    BoolMatching = scoH[0]==scoH[1]
    
    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]
    if numberContrastive =="batch_size":
        numberContrastive = bs
    else:
        numberContrastive = int(numberContrastive)
        
    lossMatrix = torch.zeros((bs,numberContrastive)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([numberContrastive-1]*bs).to(device)
    for i in range(bs):
        if parcimony ==False:
            #print("makesquared", i)
            idx_input = idx_list[i]
            inp_repeted = inp_data[:,i,:].unsqueeze(1).repeat(1, numberContrastive, 1)
            # print(i, inp_repeted.shape) 
            idx_output = negativesampler(scoreHungarian[idx_input, :], idx_input, numberContrastive)
                
            contrastivebatch = getPreciseBatch(pds, idx_output)
            #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))
            output = model(inp_repeted, contrastivebatch[1][:-1, :])
            output = output.reshape(-1, output.shape[2])
            _, targets_Original = contrastivebatch[1].max(dim=2)
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterion(output, targets_Original).reshape(-1,numberContrastive).mean(dim=0)
            lossMatrix[i,:] = loss
            LossCE += loss[numberContrastive-1]
            if accumulate:
                lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
                Totloss = ((1-alpha)*loss[numberContrastive-1] + alpha*lossMatchingtemp)/bs
                Totloss.backward()
                lossMatching += lossMatchingtemp.clone().detach()/bs
        else:
            idx_input = idx_list[i]
            if BoolMatching[idx_input]==True:
                inp_repeted = inp_data[:,i,:].unsqueeze(1)
                
                # print(i, inp_repeted.shape) 
                #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))
                output = model(inp_repeted, target[:,i,:].unsqueeze(1)[:-1, :])
                output = output.reshape(-1, output.shape[2])
                _, targets_Original = target[:,i,:].unsqueeze(1).max(dim=2)
                targets_Original = targets_Original[1:].reshape(-1)
                loss = criterion(output, targets_Original).reshape(-1,1).mean(dim=0)
                lossMatrix[i,:] = loss
                LossCE += loss[0]
                if accumulate:
                    Totloss = ((1-alpha)*loss[0])/bs
                    Totloss.backward()
                    lossMatching += 0.0
            else:
                idx_input = idx_list[i]
                inp_repeted = inp_data[:,i,:].unsqueeze(1).repeat(1, numberContrastive, 1)
                # print(i, inp_repeted.shape) 
                idx_output = negativesampler(scoreHungarian[idx_input, :], idx_input, numberContrastive)
                    
                contrastivebatch = getPreciseBatch(pds, idx_output)
                #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))
                output = model(inp_repeted, contrastivebatch[1][:-1, :])
                output = output.reshape(-1, output.shape[2])
                _, targets_Original = contrastivebatch[1].max(dim=2)
                targets_Original = targets_Original[1:].reshape(-1)
                loss = criterion(output, targets_Original).reshape(-1,numberContrastive).mean(dim=0)
                lossMatrix[i,:] = loss
                LossCE += loss[numberContrastive-1]
                if accumulate:
                    lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
                    Totloss = ((1-alpha)*loss[numberContrastive-1] + alpha*lossMatchingtemp)/bs
                    Totloss.backward()
                    lossMatching += lossMatchingtemp.clone().detach()/bs
                
    if accumulate==False:        
        LossCE /=bs
        lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
        lossMatching = criterionMatching(lossMatrix, targetMatching)# + 0.5*criterionMatching(torch.t(lossMatrix), targetMatching)
        return LossCE, lossMatching
    else:
        LossCE /=bs
        return LossCE.clone().detach(), lossMatching



def HardNegativeSamplingContrastiveMatchingLossTest(batch,
                                                model,
                                                scoreHungarian,
                                                pds,
                                                criterion,
                                                criterionMatching,
                                                device,
                                                accumulate=False,
                                                alpha=0.0,
                                                numberContrastive="batch_size",
                                                negativesampler=MinHardNegative):
    
    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]
    if numberContrastive =="batch_size":
        numberContrastive = bs
    else:
        numberContrastive = int(numberContrastive)
    lossMatrix = torch.zeros((bs,numberContrastive)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([numberContrastive-1]*bs).to(device)
    for i in range(bs):
        #print("makesquared", i)
        idx_input = idx_list[i]
        if idx_input ==0:
            inp_repeted = inp_data[:,i,:].unsqueeze(1).repeat(1,bs,1)
            # print(i, inp_repeted.shape)
            idx_output = negativesampler(scoreHungarian[idx_input, :], idx_input, numberContrastive)
            if idx_input ==0:
    
                print("idxoutput", idx_output)
                print("check", idx_input, idx_output[0])
                losscheck = checklossContrastive(model, pds, 0, idx_output[0])
                losscheckTrue = checklossContrastive(model, pds, 0, 0)
                print(losscheck, scoreHungarian[idx_input, idx_output[0]], losscheckTrue, scoreHungarian[idx_input, idx_input])
                
            contrastivebatch = getPreciseBatch(pds, idx_output)
            #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))
            output = model(inp_repeted, contrastivebatch[1][:-1, :])
            output = output.reshape(-1, output.shape[2])
            _, targets_Original = contrastivebatch[1].max(dim=2)
            targets_Original = targets_Original[1:].reshape(-1)
            loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
            lossMatrix[i,:] = loss
            LossCE += loss[numberContrastive-1]
    
            
            print(i, "CEloss", loss)
            print("target", targetMatching[i])
            print("loss for target", [float(criterionMatching((-1*loss).unsqueeze(0), torch.tensor(itest).unsqueeze(0).to(device))) for itest in range(numberContrastive)])
                
            if accumulate:
                lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
                Totloss = ((1-alpha)*loss[numberContrastive-1] + alpha*lossMatchingtemp)/bs
                Totloss.backward()
                lossMatching += lossMatchingtemp.clone().detach()/bs
    if accumulate==False:        
        LossCE /=bs
        lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
        lossMatching = criterionMatching(lossMatrix, targetMatching)# + 0.5*criterionMatching(torch.t(lossMatrix), targetMatching)
        return LossCE, lossMatching
    else:
        LossCE /=bs
        return LossCE.clone().detach(), lossMatching





def PointwiseMutualInformationLoss(batch,
                                   model,
                                   scoreHungarian,
                                   pds,
                                   criterion,
                                   criterionMatching,
                                   device,
                                   accumulate=False,
                                   alpha=0.0,
                                   numberContrastive="batch_size",
                                   negativesampler=MinHardNegative):
    

    
    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]
    if numberContrastive =="batch_size":
        numberContrastive = bs
    else:
        numberContrastive = int(numberContrastive)
        
    lossMatrix = torch.zeros((bs,numberContrastive)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([numberContrastive-1]*bs).to(device)
    for i in range(bs):
        #print("makesquared", i)
        idx_input = idx_list[i]
        
        
    #inp_repeted = inp_data[:,i,:].unsqueeze(1).repeat(1,bs,1)
        out_repeted = target[:,i,:].unsqueeze(1).repeat(1,numberContrastive,1)
        # print(i, inp_repeted.shape)
        idx_output = negativesampler(scoreHungarian[:, idx_input], idx_input, numberContrastive)
            
        contrastivebatch = getPreciseBatch(pds, idx_output)
        #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))
        # output = model(inp_repeted, contrastivebatch[1][:-1, :])
        output = model(contrastivebatch[0], out_repeted[:-1, :])
        
        output = output.reshape(-1, output.shape[2])
        _, targets_Original = out_repeted.max(dim=2)
        targets_Original = targets_Original[1:].reshape(-1)
        loss = criterion(output, targets_Original).reshape(-1,numberContrastive).mean(dim=0)
        lossMatrix[i,:] = loss
        LossCE += loss[numberContrastive-1]
        if accumulate:
            lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
            Totloss = ((1-alpha)*loss[numberContrastive-1] + alpha*lossMatchingtemp)/bs
            Totloss.backward()
            lossMatching += lossMatchingtemp.clone().detach()/bs

    if accumulate==False:        
        LossCE /=bs
        lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
        lossMatching = criterionMatching(lossMatrix, targetMatching)# + 0.5*criterionMatching(torch.t(lossMatrix), targetMatching)
        return LossCE, lossMatching
    else:
        LossCE /=bs
        return LossCE.clone().detach(), lossMatching
    
    

def SamplerContrastiveMatchingLoss(batch,
                                    model,
                                    criterion,
                                    criterionMatching,
                                    device,
                                    accumulate=False,
                                    alpha=0.0,
                                    numberContrastive="batch_size",
                                    sampler="gumbel"):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]
    if numberContrastive =="batch_size":
        numberContrastive = bs
    else:
        numberContrastive = int(numberContrastive)
        
    lossMatrix = torch.zeros((bs,numberContrastive)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([numberContrastive-1]*bs).to(device)
    for i in range(bs):
        idx_input = idx_list[i]
        inp_repeted = inp_data[:,i].unsqueeze(1).repeat(1, numberContrastive)
        # print("1",inp_data.shape, inp_data[:,i].unsqueeze(1).shape, inp_repeted.shape)
        targi = torch.nn.functional.one_hot(target[:,i].unsqueeze(1), num_classes=model.trg_vocab_size)
        #print("targi",targi.shape)
        # idx_output = negativesampler(scoreHungarian[idx_input, :], idx_input, numberContrastive)
        # contrastivebatch = getPreciseBatch(pds, idx_output)
        contrastiveTarget = model.pseudosample(inp_data[:,i].unsqueeze(1), target[:,i].unsqueeze(1), nsample=numberContrastive-1, method=sampler)
       # print("1",contrastiveTarget.shape)
        #contrastiveTarget = contrastiveTarget.max(dim=2)[1]
        contrastiveTarget = torch.cat([contrastiveTarget, targi], dim=1)
        #print("check is getting the right", torch.equal(contrastivebatch[0][:,numberContrastive-1,:], inp_data[:,i,:]))

        output = model(inp_repeted, contrastiveTarget[:-1, :])
        output = output.reshape(-1, output.shape[2])
        _, targets_Original = contrastiveTarget.max(dim=2)
        targets_Original = targets_Original[1:].reshape(-1)
        loss = criterion(output, targets_Original).reshape(-1,numberContrastive).mean(dim=0)
        #loss = 
        lossMatrix[i,:] = loss
        LossCE += loss[numberContrastive-1]
        if accumulate:
            lossMatchingtemp = criterionMatching((-1*loss).unsqueeze(0), targetMatching[i].unsqueeze(0))
            #lossMatchingtemp = loss[numberContrastive-1] - torch.mean(loss[:-2])
            Totloss = ((1-alpha)*loss[numberContrastive-1] + alpha*lossMatchingtemp)/bs
            Totloss.backward()
            lossMatching += lossMatchingtemp.clone().detach()/bs
        
                
    if accumulate==False:        
        LossCE /=bs
        lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
        lossMatching = criterionMatching(lossMatrix, targetMatching)# + 0.5*criterionMatching(torch.t(lossMatrix), targetMatching)
        return LossCE, lossMatching
    else:
        LossCE /=bs
        return LossCE.clone().detach(), lossMatching
    
    
    

def ConditioalEntropyMatchingLoss(batch,
                                  model,
                                  CCL_mean,
                                  device,
                                  samplingMultiple=1,
                                  gumbel=True):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    output = model(inp_data, target[:-1, :])
    acc = accuracy(batch, output, onehot=model.onehot).item()
    output = output.reshape(-1, output.shape[2])#keep last dimension
    if model.onehot:
        _, targets_Original = target.max(dim=2)
    else:
        targets_Original= target
    
    targets_Original = targets_Original[1:].reshape(-1)
    lossCE = CCL_mean(output, targets_Original)
    
    
    ## Entropic 
    samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
    ### fake step
    if gumbel==False:
        samples = samples.max(dim=2)[1]
    output = model(inp_data, samples[:-1, :])
    output = output.reshape(-1, output.shape[2])

    if gumbel ==True:
        _, samples_Original = samples.max(dim=2)
    else: 
        samples_Original = samples
    samples_Original = samples_Original[1:].reshape(-1)
    lossEntropy = CCL_mean(output, samples_Original)
    for i in range(samplingMultiple-1):
        samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
        if gumbel==False:
            samples = samples.max(dim=2)[1]
        output = model(inp_data, samples[:-1, :])
        output = output.reshape(-1, output.shape[2])
        if gumbel ==True:
            _, samples_Original = samples.max(dim=2)
        else: 
            samples_Original = samples
        samples_Original = samples_Original[1:].reshape(-1)
        lossEntropy += CCL_mean(output, samples_Original)
    
    lossEntropy = lossEntropy/samplingMultiple  

    
    return lossCE, lossEntropy, acc





def ConditionalSquaredEntropyMatchingLoss(batch,
                                  model,
                                  CCL_mean,
                                  device,
                                  samplingMultiple=1,
                                  gumbel=True):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    output = model(inp_data, target[:-1, :])
    acc = accuracy(batch, output, onehot=model.onehot).item()
    output = output.reshape(-1, output.shape[2])#keep last dimension
    if model.onehot:
        _, targets_Original = target.max(dim=2)
    else:
        targets_Original= target
    
    targets_Original = targets_Original[1:].reshape(-1)
    lossCE = CCL_mean(output, targets_Original)
    
    
    ## Entropic 
    samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
    ### fake step
    if gumbel==False:
        samples = samples.max(dim=2)[1]
    output = model(inp_data, samples[:-1, :])
    output = output.reshape(-1, output.shape[2])

    if gumbel ==True:
        _, samples_Original = samples.max(dim=2)
    else: 
        samples_Original = samples
    samples_Original = samples_Original[1:].reshape(-1)
    lossEntropy = -1*torch.exp(-1*CCL_mean(output, samples_Original))
    for i in range(samplingMultiple-1):
        samples = model.pseudosample(inp_data, target, nsample=1, method="gumbel")
        if gumbel==False:
            samples = samples.max(dim=2)[1]
        output = model(inp_data, samples[:-1, :])
        output = output.reshape(-1, output.shape[2])
        if gumbel ==True:
            _, samples_Original = samples.max(dim=2)
        else: 
            samples_Original = samples
        samples_Original = samples_Original[1:].reshape(-1)
        lossEntropy += torch.exp(-1*CCL_mean(output, samples_Original))
    
    lossEntropy = lossEntropy/samplingMultiple  

    
    return lossCE, lossEntropy, acc
 
    
# for batch_idx, batch in enumerate(train_iterator):
#     inp_data, target= batch[0], batch[1]
#     output = model(inp_data, target[:-1, :])
#     output = output.reshape(-1, output.shape[2])#keep last dimension
#     _, targets_Original = target.max(dim=2)
#     targets_Original = targets_Original[1:].reshape(-1)
#     optimizer.zero_grad()
#     loss = criterion(output, targets_Original)
#     losses.append(loss.item())
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
#     optimizer.step()





def SamplerContrastiveMatchingLossBin(batch,
                                    model,
                                    criterion,
                                    criterionMatching,
                                    device,
                                    accumulate=False,
                                    sampler="gumbel"):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]

        
    lossMatrix = torch.zeros((bs,2)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([0]*bs).to(device)
    
    contrastiveTarget = model.pseudosample(inp_data, target, nsample=1, method=sampler)
    
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])
    targets_Original = target
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,0] = loss
    LossCE += loss.mean()
    
    output2 = model(inp_data, contrastiveTarget[:-1, :])
    output2 = output2.reshape(-1, output2.shape[2])
    _, targets_Original2 = contrastiveTarget.max(dim=2)
    targets_Original2 = targets_Original2[1:].reshape(-1)
    loss2 = criterion(output2, targets_Original2).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,1] = loss2
    lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
    lossMatching = criterionMatching(lossMatrix, targetMatching)

    return LossCE, lossMatching
    

def ReyniMatchingLossBin(batch,
                                    model,
                                    criterion,
                                    device,
                                    accumulate=False,
                                    sampler="gumbel"):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]

        
    lossMatrix = torch.zeros((bs,2)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([0]*bs).to(device)
    
    contrastiveTarget = model.pseudosample(inp_data, target, nsample=1, method=sampler)
    
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])
    targets_Original = target
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,0] = loss
    LossCE += loss.mean()
    
    output2 = model(inp_data, contrastiveTarget[:-1, :])
    output2 = output2.reshape(-1, output2.shape[2])
    _, targets_Original2 = contrastiveTarget.max(dim=2)
    targets_Original2 = targets_Original2[1:].reshape(-1)
    loss2 = criterion(output2, targets_Original2).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,1] = loss2
    lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
    lossMatching = criterionMatching(lossMatrix, targetMatching)

    return LossCE, lossMatching


def ReyniMatchingLossNew(batch,
                                    model,
                                    criterion,
                                    criterionMatching,
                                    device,
                                    accumulate=False,
                                    ncontrastive=5,
                                    sampler="gumbel"):

    inp_data, target, idx_list = batch[0], batch[1], batch[2]
    bs = inp_data.shape[1]

        
    lossMatrix = torch.zeros((bs,ncontrastive+1)).to(device)
    LossCE = torch.tensor(0.0).to(device)
    lossMatching = torch.tensor(0.0).to(device)
    targetMatching = torch.tensor([0]*bs).to(device)
    
    
    
    output = model(inp_data, target[:-1, :])
    output = output.reshape(-1, output.shape[2])
    targets_Original = target
    targets_Original = targets_Original[1:].reshape(-1)
    loss = criterion(output, targets_Original).reshape(-1,bs).mean(dim=0)
    lossMatrix[:,0] = loss
    LossCE += loss.mean()
    
    for i in range(1,ncontrastive+1):
        contrastiveTarget = model.pseudosample(inp_data, target, nsample=1, method=sampler)
        output2 = model(inp_data, contrastiveTarget[:-1, :])
        output2 = output2.reshape(-1, output2.shape[2])
        _, targets_Original2 = contrastiveTarget.max(dim=2)
        targets_Original2 = targets_Original2[1:].reshape(-1)
        loss2 = criterion(output2, targets_Original2).reshape(-1,bs).mean(dim=0)
        lossMatrix[:,i] = loss2
        
    lossMatrix *=-1# torch.nn.functional.softmax(lossMatrix, dim=0)
    lossMatching = criterionMatching(lossMatrix, targetMatching)

    return LossCE, lossMatching


