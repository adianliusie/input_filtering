import torch
import random

from typing import List, Tuple
from collections.abc import Iterator
from types import SimpleNamespace

class ProbeBatcher():
    def batches(self, h:dict, labels:dict, bsz:int, shuffle:bool=False)->Iterator:
        """splits the data into batches and returns them"""
        data = self._prep_data(h, labels)
        if shuffle: random.shuffle(data)
        batches = [data[i:i+bsz] for i in range(0,len(data), bsz)]
        for batch in batches:
            yield self.batchify(batch)
  
    def batchify(self, batch:List[list])->SimpleNamespace:
        """each input is input ids and mask for utt, + label"""
        h, labels = zip(*batch)  
        h = torch.FloatTensor(h)
        labels = torch.LongTensor(labels)
        return SimpleNamespace(h=h, labels=labels)
    
    def _prep_data(self, h_dict:dict, labels_dict:dict)->Tuple[list, list]:
        assert h_dict.keys() == labels_dict.keys()
        keys = sorted(list(h_dict.keys()))
        H = [h_dict[k] for k in keys]
        labels = [labels_dict[k] for k in keys]
        return list(zip(H, labels))
    
    @staticmethod
    def is_pred_different(preds_1, preds_2)->dict:
        """ 0 for all ids that have the same prediction, otherwise 1"""
        assert preds_1.keys() == preds_2.keys()
        diff_labels = {k: preds_1[k] != preds_2[k] for k in preds_1.keys()}
        return diff_labels

    def __call__(self, *args, **kwargs):
        return self.batches(*args, **kwargs)
    