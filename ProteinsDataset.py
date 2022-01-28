#-*- coding: utf-8 -*-

###### 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math 
import numpy as np
import pandas as pd
# from utils import *
from torch._six import string_classes
import collections
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

def default_collate(batch):
    r"""Puts each data field into a tensor with first dimension batch size. 
    Modified to get 1st dim as batch dim"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 1, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def getUnique(tensor):
    inverseMapping = torch.unique(tensor, dim=1, return_inverse=True)[1]
    dic=defaultdict(lambda:0)
    BooleanKept = torch.tensor([False] * tensor.shape[1])
    for i in range(tensor.shape[1]):
        da = int(inverseMapping[i])
        if dic[da]==0:
            # if da ==47:
            #     print(da, dic[da])
            BooleanKept[i]=True
        dic[da] +=1
    return tensor[:, BooleanKept, :], BooleanKept


class ProteinTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, csvPath,  mapstring = "-ACDEFGHIKLMNPQRSTVWY", transform=None, device=None, batch_first=False, Unalign=False, filteringOption='none', returnIndex=False, onehot=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csvPath, header=None)
        self.q=len(mapstring);
        self.SymbolMap=dict([(mapstring[i],i) for i in range(len(mapstring))])
        self.init_token = "<sos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk="X"
        self.mapstring = mapstring
        self.onehot=onehot
        self.SymbolMap=dict([(mapstring[i],i) for i in range(len(mapstring))])
        self.SymbolMap[self.unk] = len(mapstring)
        self.SymbolMap[self.init_token] = len(mapstring)+1
        self.SymbolMap[self.eos_token] = len(mapstring)+2
        self.SymbolMap[self.pad_token] = len(mapstring)+3
        self.inputsize = len(df.iloc[1][0].split(" "))+2
        self.outputsize = len(df.iloc[1][1].split(" "))+2
        self.gap = "-"
        self.tensorIN=torch.zeros(self.inputsize,len(df), len(self.SymbolMap))
        self.tensorOUT=torch.zeros(self.outputsize,len(df), len(self.SymbolMap))
        self.device = device
        self.transform = transform
        self.batch_first = batch_first
        self.filteringOption = filteringOption
        self.returnIndex = returnIndex
        if Unalign==False:
            print("keeping the gap")
            for i in range(len(df)):
                inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[0][i].split(" ")]+[self.SymbolMap[self.eos_token]]
                out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[1][i].split(" ")]+[self.SymbolMap[self.eos_token]]
                self.tensorIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(out), num_classes=len(self.SymbolMap))
        else:
            print("Unaligning and Padding")
            for i in range(len(df)):
                inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[0][i].split(" ") if k!=self.gap]+[self.SymbolMap[self.eos_token]]
                out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k] for k in df[1][i].split(" ") if k!=self.gap]+[self.SymbolMap[self.eos_token]]
                inp += [self.SymbolMap[self.pad_token]]*(self.inputsize - len(inp))
                out += [self.SymbolMap[self.pad_token]]*(self.outputsize - len(out))
                self.tensorIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(out), num_classes=len(self.SymbolMap))
                
        if filteringOption == "in":
            a = getUnique(self.tensorIN)[1]
            self.tensorIN = self.tensorIN[:,a,:]
            self.tensorOUT = self.tensorOUT[:,a,:]
            print("filtering the redundancy of input proteins")
        elif filteringOption == "out":
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:,b,:]
            self.tensorOUT = self.tensorOUT[:,b,:]
            print("filtering the redundancy of output proteins")
        elif filteringOption == "and":
            a = getUnique(self.tensorIN)[1]
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:,a*b,:]
            self.tensorOUT = self.tensorOUT[:,a*b,:]
            print("filtering the redundancy of input AND output proteins")
        elif filteringOption == "or":
            a = getUnique(self.tensorIN)[1]
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:,a+b,:]
            self.tensorOUT = self.tensorOUT[:,a+b,:]
            print("filtering the redundancy of input OR output proteins")
        else:
            print("No filtering of redundancy")
            
        if batch_first:
            self.tensorIN = torch.transpose(self.tensorIN, 0,1)
            self.tensorOUT = torch.transpose(self.tensorOUT, 0,1)
        
        if onehot==False:
            self.tensorIN = self.tensorIN.max(dim=2)[1]
            self.tensorOUT = self.tensorOUT.max(dim=2)[1]
            

        
            
        if device != None:
            self.tensorIN= self.tensorIN.to(device, non_blocking=True)
            self.tensorOUT= self.tensorOUT.to(device, non_blocking=True)
            
            

    def __len__(self):
        if self.batch_first:
            return self.tensorIN.shape[0]
        else:
            return self.tensorIN.shape[1]

    def __getitem__(self, idx): # from the dataset, gives the data in the form it will be used by the NN
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.batch_first:
            if self.returnIndex:
                return self.tensorIN[idx,:], self.tensorOUT[idx,:], idx
            else:
                return self.tensorIN[idx,:], self.tensorOUT[idx,:]
        else:
            if self.returnIndex:
                return self.tensorIN[:,idx], self.tensorOUT[:,idx], idx
            else:
                return self.tensorIN[:,idx], self.tensorOUT[:,idx]
        
    def to(self,device):
        if device != None:
            self.tensorIN= self.tensorIN.to(device, non_blocking=True)
            self.tensorOUT= self.tensorOUT.to(device, non_blocking=True)
            
    ## TO DO join for onehot=False
    def join(self, pds):
        if self.device != pds.device:
            pds.tensorIN= pds.tensorIN.to(self.device, non_blocking=True)
            pds.tensorOUT= pds.tensorOUT.to(self.device, non_blocking=True)
        if self.onehot:
            if self.inputsize < pds.inputsize:
                dif = pds.inputsize - self.inputsize
                padIN = torch.zeros(dif, len(self), len(self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(len(self)):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
                self.inputsize = pds.inputsize
            elif self.inputsize > pds.inputsize:
                dif = self.inputsize - pds.inputsize
                padIN = torch.zeros(dif, len(pds), len(self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(len(pds)):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
                pds.inputsize = self.inputsize
            if self.outputsize < pds.outputsize:
                dif = pds.outputsize - self.outputsize
                padOUT = torch.zeros(dif, self.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(self.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
                self.outputsize = pds.outputsize
            elif self.outputsize > pds.outputsize:
                dif = self.outputsize - pds.outputsize
                padOUT = torch.zeros(dif, pds.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(pds.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
                pds.outputsize = self.outputsize
        else:
            if self.inputsize < pds.inputsize:
                dif = pds.inputsize - self.inputsize
                padIN = torch.zeros(dif, len(self)).to(self.device, non_blocking=True)
                for i in range(len(self)):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padIN[:,i] = torch.tensor(inp)
                self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
                self.inputsize = pds.inputsize
            elif self.inputsize > pds.inputsize:
                dif = self.inputsize - pds.inputsize
                padIN = torch.zeros(dif, len(pds)).to(self.device, non_blocking=True)
                for i in range(len(pds)):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padIN[:,i] = torch.tensor(inp)
                self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
                pds.inputsize = self.inputsize
            if self.outputsize < pds.outputsize:
                dif = pds.outputsize - self.outputsize
                padOUT = torch.zeros(dif, self.tensorOUT.shape[1]).to(self.device, non_blocking=True)
                for i in range(self.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padOUT[:,i] = torch.tensor(inp)
                self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
                self.outputsize = pds.outputsize
            elif self.outputsize > pds.outputsize:
                dif = self.outputsize - pds.outputsize
                padOUT = torch.zeros(dif, pds.tensorOUT.shape[1]).to(self.device, non_blocking=True)
                for i in range(pds.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padOUT[:,i] = torch.tensor(inp)
                self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
                pds.outputsize = self.outputsize
            
            
def getPreciseBatch(pds, idxToget):
    data = []
    for idx in idxToget:
        data.append(pds[idx])
    batch = default_collate(data)
    return batch


# def OneHot(in_tensor):
#     seq_length, N = in_tensor.shape
#     out_one_hot = torch.zeros((in_tensor.shape[0], in_tensor.shape[1],len(self.SymbolMap)))
#     for i in range(seq_length):
#         for j in range(N):
#             c = in_tensor[i,j]
#             out_one_hot[i,j,c] = 1
#     return out_one_hot



def getBooleanisRedundant(tensor1, tensor2):
    l1 = tensor1.shape[1]
    l2 = tensor2.shape[1]
    BooleanKept = torch.tensor([True]*l2)
    for i in range(l1):
        protein1 = tensor1[:,i,:]
        for j in range(l2):
            protein2 = tensor2[:,j,:]
            if torch.equal(protein1, protein2):
                BooleanKept[j]=False
    return BooleanKept
    


def deleteRedundancyBetweenDatasets(pds1, pds2):

    filteringOption = pds1.filteringOption
    if filteringOption == "in":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        pds2.tensorIN = pds2.tensorIN[:,a,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a,:]
    elif filteringOption == "out":
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,b,:]
    elif filteringOption == "and":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,a*b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a*b,:]
    elif filteringOption == "or":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:,a+b,:]
        pds2.tensorOUT = pds2.tensorOUT[:,a+b,:]



            
            
    
            





        









# =============================================================================
# 
# from typing import Optional, Dict, List, Tuple
# from Bio import SeqIO
# import torch
# from torch import Tensor
# import numpy as np
# import logging
# import copy
# from sklearn.model_selection import train_test_split

# log = logging.getLogger(__name__)
# # =============================================================================
# 
# 
# class AminoAcidMap:
#     """ Class representing a mapping between amino acids and integer indices.
# 
#     Keyword Args:
#         mapstring(str): String of symbols; contiguous integer ids are assigned to amino acids in the
#                         order they appear in the mapstring. Special indices (see other arguments)
#                         are jumped and not assigned to amino acids.
#                         Default \"ARNDCEQGHILKMFPSTWYV-\".
# 
#         ignore_lowercase(bool): Ignore lowercase characters when translating sequences to numeric
# 
#         ignore_characters(str): Characters in this string are ignored when translating sequences to
#                                 numeric
# 
#         SOS_INDEX(Optional[int]): If set, numeric sequences start with SOS_INDEX.
#                                   Default: None.
# 
#         EOS_ID(Optional[int]): If set, numeric sequences end with EOS_INDEX.
#                                Default: None.
# 
#         UNKNOWN_INDEX(Optional[int]): If set , unknown symbols are mapped to UNKNOWN_INDEX.
#                                       Default: None.
# 
#         UNKNOWN_AA(Optional[str]): If set, unknown amino acids are mapped to UNKOWN_AA. This
#                                    is mutually exclusive with UNKNOWN_INDEX. Default: \"-\".
#     """
# 
#     def __init__(self, mapstring: str = "ARNDCEQGHILKMFPSTWYV-",
#                  ignore_lowercase: bool = True,
#                  ignore_characters: str = ".",
#                  SOS_INDEX: Optional[int] = None,
#                  EOS_INDEX: Optional[int] = None,
#                  UNKNOWN_INDEX: Optional[int] = None,
#                  UNKNOWN_AA: Optional[str] = "-"):
#         """ See class documentation """
# 
#         # check arguments
#         if len(mapstring) < 1:
#             log.error("mapstring must have length > 0")
# 
#         self.mapstring: str = mapstring
#         self.ignore_lowercase: bool = ignore_lowercase
#         self.ignore_characters: str = ignore_characters
# 
#         self.SOS_INDEX: Optional[int] = SOS_INDEX
#         self.EOS_INDEX: Optional[int] = EOS_INDEX
# 
#         if UNKNOWN_INDEX is not None and UNKNOWN_AA is not None:
#             log.error("cannot set UNKNOWN_ID and UNKNOWN_AA at the same time")
# 
#         if UNKNOWN_AA and UNKNOWN_AA not in mapstring:
#             log.error("UNKNOWN_AA must be in mapstring")
# 
#         self.UNKNOWN_AA: Optional[str] = UNKNOWN_AA
#         self.UNKNOWN_INDEX: Optional[int] = UNKNOWN_INDEX
# 
#         self.aa2index: Dict[str, int] = dict()
# 
#         aa_index: int = 0
#         reserved_indices: List[int] = [reserved_index for reserved_index in
#                                        [UNKNOWN_INDEX, SOS_INDEX, EOS_INDEX] if reserved_index]
#         for aa_char in mapstring:
#             while aa_index in reserved_indices:
#                 aa_index += 1
#             self.aa2index[aa_char] = aa_index
#             aa_index += 1
# 
#         if min(*reserved_indices, *self.aa2index.values()) != 0:
#             log.error("minimal index is not 0; this should not happen")
# 
#         self.max_index: int = max(*reserved_indices, *self.aa2index.values())
#         self.n_symbols: int = self.max_index + 1
# 
#         log.info("mapping to indices 0:{}".format(self.max_index))
# 
#         if self.n_symbols > len(self.aa2index) + len(reserved_indices):
#             log.warning("index range is larger than known symbols - special indices\
#                         (UNKNOWN_ID, SOS_INDEX, EOS_INDEX) are probably too high")
# 
#     def translate_string(self, seqstring: str) -> list:
#         """ translates amino acid character sequence and returns numeric sequence
# 
#         Args:
#             seqstring(str): Amino acid string to be translated
#         """
# 
#         seq_numeric: list = [self.SOS_INDEX] if self.SOS_INDEX else []
#         for aa in seqstring:
#             if self.ignore_lowercase and aa.islower():
#                 continue
#             if aa in self.ignore_characters:
#                 continue
#             if aa in self.aa2index:
#                 seq_numeric.append(self.aa2index[aa])
#             elif self.UNKNOWN_AA:
#                 seq_numeric.append(self.aa2index[self.UNKNOWN_AA])
#             elif self.UNKNOWN_INDEX:
#                 seq_numeric.append(self.UNKNOWN_INDEX)
#             else:
#                 log.error("cannot deal with aa {}".format(aa))
# 
#         if self.EOS_INDEX:
#             seq_numeric.append(self.EOS_INDEX)
# 
#         return seq_numeric
# 
# 
# class AminoAcidSequence(object):
#     """ Class representing one amino acid sequence; wrapper around SeqIO.SeqRecord
#         If amino acid map is available, sequences are mapped to index sequences
#         automatically.
# 
#     Args:
#         seq_record(SeqIO.SeqRecord): SeqIO.SeqRecord to be wrapped
# 
#     Keyword Args:
#         amino_acid_map(Optional[AminoAcidMap]): Amino acid map for mapping to index sequence
#                                                 Default: AminoAcidMap()
#     """
# 
#     def __init__(self, seq_record: SeqIO.SeqRecord,
#                  amino_acid_map: Optional[AminoAcidMap] = AminoAcidMap()):
# 
#         self.seq_record: SeqIO.SeqRecord = seq_record
#         self.seq_numeric: Optional[List[int]] = None
#         if amino_acid_map:
#             self.seq_numeric = amino_acid_map.translate_string(seq_record.seq)
# 
# 
# class AminoAcidSequences(object):
# 
#     """ Class representing sets of amino acid sequences; if filepath is given, parsing is done
#         automatically. If amino_acid_map is available sequences are translated to index sequences.
#         Sequences are stored in the \"sequences\" dictionary, which maps from the \"id\" file of the
#         SeqIO.SeqRecord object to the SeqRecord object.
# 
#         Keyword Args:
#             amino_acid_map(Optional[AminoAcidMap]): Amino acid map for mapping to index sequences
#                                                     Default: AminoAcidMap()
# 
#             filepath(Optional[str]): file path to parse
#                                      Default: None
# 
#             format(Optional[str]): format of file
#                                    Default: \"fasta\"
# 
#             same_length(bool): check that numeric sequences have same length after
#                                 discarding ignored characters.
#                                 Default: True
#     """
# 
#     def __init__(self, filepath: Optional[str] = None,
#                  amino_acid_map: AminoAcidMap = AminoAcidMap(),
#                  format: Optional[str] = "fasta",
#                  same_length: bool = True):
# 
#         """ See class documentation """
# 
#         self.filepath: Optional[str] = filepath
#         self.format: Optional[str] = format
#         self.amino_acid_map: Optional[AminoAcidMap] = amino_acid_map
#         self.sequences: Optional[Dict[str, AminoAcidSequence]] = None
#         self.sequence_length: Optional[int] = None
#         self.same_length: bool = same_length
# 
#         if not self.filepath:
#             log.warning("no filepath set; not parsing anything.")
#         elif self.filepath and self.format:
#             self.parse_from_filepath()
#             if self.sequences is None:
#                 raise ValueError("could not parse file")
#             if amino_acid_map is not None:
#                 if self.same_length and not self.check_and_set_same_length():
#                     raise ValueError("not all sequences have same length or are None")
# 
#     def check_and_set_same_length(self, report_uniques: bool = True):
#         """ Check if all numeric sequences have same length. Returns True if all are
#             same length and no sequence is None. Reports examples if sequences have
#             different lengths.
#         """
# 
#         if self.sequences is None:
#             raise ValueError("cannot check for same length sequences if sequnces is None")
# 
#         # map length -> exemplar
#         length2sequence: Dict[Optional[int], AminoAcidSequence] = \
#             {(len(sequence.seq_numeric) if sequence.seq_numeric else None): sequence
#              for sequence in self.sequences.values()}
# 
#         if None in length2sequence.values():
#             log.warning("At least 1 sequence length is None")
#             return False
# 
#         if len(length2sequence.keys()) == 1:
#             self.sequence_length = list(length2sequence.keys())[0]
#             log.info("all sequences have the same length {}".format(self.sequence_length))
#             return True
#         else:
#             log.warning("sequences have different lengths, e.g.:")
#             for (length, sequence) in length2sequence.items():
#                 log.warning("{}: {}".format(sequence.seq_record.id, length))
#             return False
# 
#     def parse_from_filepath(self):
# 
#         """ Parse AminoAcidSequences to SeqRecords and store in self.sequences """
# 
#         self.sequences: Dict[str, AminoAcidSequence] = {}
# 
#         for seq_record in SeqIO.parse(self.filepath, self.format):
#             amino_acid_sequence = AminoAcidSequence(seq_record, amino_acid_map=self.amino_acid_map)
#             self.sequences[seq_record.id] = amino_acid_sequence
# 
#     def get_batch(self, device: torch.device, batch_size: Optional[int] = 128) -> Tensor:
# 
#         if self.sequences is None:
#             raise ValueError("Tried to get batch but sequences is None")
# 
#         input_batch: List[AminoAcidSequence]
#         if batch_size is not None:
#             input_batch = np.random.choice(list(self.sequences.values()), batch_size)
#         else:
#             input_batch = list(self.sequences.values())
# 
#         input_batch_as_list = [torch.tensor(seq.seq_numeric,
#                                device=device) for seq in input_batch]
# 
#         return torch.stack(input_batch_as_list)
# 
#     def train_test_split(self, **kwargs) \
#             -> Tuple['AminoAcidSequences', 'AminoAcidSequences']:
#         """ split into train/test. uses sklearn.model_selection.train_test_split
#             and kwards are passed through. Returned values are shallow copies except
#             for sequences dict.
# 
#             Args:
#                 same as sklearn.model_selection.train_test_split
#             Returns:
#                 split_1: First part
#                 split_2: Second part
#         """
# 
#         if self.sequences is None:
#             raise ValueError("cannot do split if no sequences are available")
# 
#         sequences_train_ids: List[str]
#         sequences_test_ids: List[str]
# 
#         sequences_train_ids, sequences_test_ids = \
#             train_test_split(list(self.sequences.keys()), **kwargs)
# 
#         # make shallow copies
#         sequences_train: AminoAcidSequences = copy.copy(self)
#         sequences_test: AminoAcidSequences = copy.copy(self)
# 
#         sequences_train.sequences = {sequence_id: self.sequences[sequence_id]
#                                      for sequence_id in sequences_train_ids}
#         sequences_test.sequences = {sequence_id: self.sequences[sequence_id]
#                                     for sequence_id in sequences_test_ids}
# 
#         return sequences_train, sequences_test
# 
# =============================================================================
