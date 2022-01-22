# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:55:43 2022

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




class MatchingScore():
    def __init__(self, pds_train, pds_val):
        self.pds_train = pds_train
        self.pds_val = pds_val
        self.masks = []
        
    def resetmasks(self):
        self.masks = []
        
    def medianSeparationMasks(self):
        dval1,dval2 = distanceTrainVal(self.pds_train, self.pds_val)
        print("median", (dval1+dval2).min(dim=0)[0].median())
        maskValclose = (dval1+dval2).min(dim=0)[0]<(dval1+dval2).min(dim=0)[0].median()
        maskValclose = maskValclose.cpu().numpy()
        self.masks.append(maskValclose)
        
        maskValfar = (dval1+dval2).min(dim=0)[0]>=(dval1+dval2).min(dim=0)[0].median()
        maskValfar = maskValfar.cpu().numpy()
        self.masks.append(maskValfar)
        
    def getMatchingScore(self, model, pds_val):
        scoreHungarianVal = HungarianMatchingBS(pds_val, model,100)
        scoHVal = scipy.optimize.linear_sum_assignment(scoreHungarianVal)
        scoreMatchingVal = sum(scoHVal[0]==scoHVal[1])
        scoreMatchingVal_detail = []
        for i in range(len(self.masks)):
            mask = self.masks[i]
            scoreMatchingVal_detail.append(sum((scoHVal[0]==scoHVal[1])[mask]))
        return scoreMatchingVal, scoreMatchingVal_detail
    
        
class TransformerLoss():
    def __init__(self, device, pad, alpha=0.0, accumulate=False, onehot=False):
        self.device = device
        self.alpha = alpha
        self.CCL_mean  = nn.CrossEntropyLoss(ignore_index=pad)
        self.CCL_pointwise = nn.CrossEntropyLoss(ignore_index=pad, reduction='none')
        self.CCL_Matching = nn.CrossEntropyLoss()
        self.accumulate = accumulate
        self.onehot = onehot
        
        
    def simpleCE(self, model, batch):
        
        inp_data, target = batch[0], batch[1]
        output = model(inp_data, target[:-1, :])
        accuracy = accuracy(batch, output, onehot=False).item()
        output = output.reshape(-1, output.shape[2])#keep last dimension
        if self.onehot:
            _, targets_Original = target.max(dim=2)
        else:
            targets_Original= target
        targets_Original = targets_Original[1:].reshape(-1)
        loss = CCL_mean(output, targets_Original)
        
        return loss, accuracy
    
    def makesquaredLoss(self, model, batch):
        return makesquaredLoss(batch, model, self.CCL_pointwise, self.CCL_Matching, self.device, accumulate=self.accumulate, alpha=self.alpha)
        
    def makeBicontrastiveLoss(self, model, batch):
        return makeBicontrastiveLoss(batch, model, self.CCL_mean, alpha=self.alpha)
    
    def makeBicontrastiveCELoss(self, model, batch):
        return makeBicontrastiveCELoss(batch, model, self.CCL_mean, alpha=self.alpha)
    
    def HardNegativeSamplingContrastiveMatchingLoss(self, model, batch, scoreHungarian, pds, parcimony=False):
        return HardNegativeSamplingContrastiveMatchingLoss(batch,
                                                       model,
                                                       scoreHungarian,
                                                       pds,
                                                       self.CCL_pointwise,
                                                       self.CCL_Matching,
                                                       self.device,
                                                       accumulate=self.accumulate,
                                                       alpha=self.alpha,
                                                       numberContrastive="batch_size",
                                                       negativesampler=MinHardNegative,
                                                       parcimony=parcimony)
    
    def PointwiseMutualInformationLoss(self, model, batch, scoreHungarian, pds):
        return PointwiseMutualInformationLoss(batch,
                                           model,
                                           scoreHungarian,
                                           pds,
                                           self.CCL_pointwise,
                                           self.CCL_Matching,
                                           self.device,
                                           accumulate=self.accumulate,
                                           alpha=self.alpha,
                                           numberContrastive="batch_size",
                                           negativesampler=MinHardNegative)
    
    
def trainTransformer(model, iterator, optimizer, loss, losstype='CE', scoreHungarian=None, pds=None):
    accuracy = 0
    lossesCE = []
    if losstype=='CE':
        for batch_idx, batch in enumerate(iterator):
            optimizer.zero_grad()
            loss, acc = loss.simpleCE(model, batch)
            accuracy+=acc
            lossesCE.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
            optimizer.step()
        mean_lossCETrain = sum(lossesCE) / len(lossesCE)
        accuracyTrain = accuracyTrain/len(iterator.dataset)
        return mean_lossCETrain, accuracyTrain
    if losstype=='HardNegativeSamplingContrastiveMatchingLoss':
        return "todo"
    
    
        
    
    

























