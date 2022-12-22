import torch
import random

from itertools import islice
from typing import List
from types import SimpleNamespace

class Batcher:
    def __init__(self, max_len:int):
        self.device     = torch.device('cpu')
        self.max_len    = max_len

    def batches(self, data:list, bsz:int, shuffle:bool=False):
        """splits the data into batches and returns them"""
        examples = self._prep_examples(data)
        if shuffle: random.shuffle(examples)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        for batch in batches:
            yield self.batchify(batch)
  
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ex_id, input_ids, labels = zip(*batch)  
        input_ids, attention_mask = self._get_padded_ids(input_ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(
            ex_id=ex_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

    def _prep_examples(self, data:list):
        """ sequence classification input data preparation"""
        prepped_examples = []
        for ex in data:
            ex_id     = ex.ex_id
            input_ids = ex.input_ids
            label     = ex.label
            
            if len(input_ids) > self.max_len:            
                input_ids = input_ids[:self.max_len-1] + [input_ids[-1]]
            prepped_examples.append([ex_id, input_ids, label])
                
        return prepped_examples
                
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    
    def _get_padded_ids(self, ids:list, pad_id=0)->List[torch.LongTensor]:
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [pad_id]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
    
class Seq2seqBatcher(Batcher):
    def _prep_examples(self, data: List) -> List[List]:
        prepped_examples = []
        for ex in data:
            ex_id = ex.ex_id
            input_ids = ex.input_ids
            label_ids = ex.label_ids
            if len(input_ids) > self.max_len or len(label_ids) > self.max_len: 
                continue
            prepped_examples.append([ex_id, input_ids, label_ids])

        return prepped_examples 

    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ex_id, input_ids, label_ids = zip(*batch)  
        input_ids, attention_mask = self._get_padded_ids(input_ids)
        label_ids, _ = self._get_padded_ids(label_ids, pad_id=-100)
        return SimpleNamespace(
            ex_id=ex_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            label_ids=label_ids
        )