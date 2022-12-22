import torch
import torch.nn.functional as F
import random
import re
import numpy as np

from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from ...handlers.evaluater import Evaluator

class CorpusAnalyser(Evaluator):
    def __init__(self, path, device='cuda'):
        # load project
        self.exp_path = path
        super().setup_helpers()
        self.model.eval()

        #set device
        self.to(device)
        self.device = device

    @torch.no_grad()
    def word_deletion_influence(self, dataset, mode, word):
        """ given a word, finds the impact of deleting the words from all examples in the corpus"""

        # select examples where word is present, and remove selected word (and double spaces)
        data_filt = self.data_handler.prep_split(dataset, mode)
        data_filt = deepcopy(data_filt)
        data_filt = [ex for ex in data_filt if word in ex.text.lower()]

        for ex in data_filt:
            ex.text = re.sub(word, '', ex.text, flags=re.IGNORECASE)
            ex.text = re.sub(' +', ' ', ex.text)

        # tokenize data and create batches
        if self.model_args.num_classes == 3: 
            raise ValueError("code only currently works for binary classification")
            # data_filt = self.data_handler._prep_ids_pairs(data_filtered)
        else: 
            data_filt = self.data_handler._prep_ids(data_filt)
        
        batches_filt = self.batcher(data=data_filt, bsz = 1,  shuffle = False)

        # calculate probabilities
        probs_filt = {}
        for batch in tqdm(batches_filt):
            ex_id = batch.ex_id[0]
            output = self.model_loss(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probs_filt[ex_id] = prob.cpu().numpy()

        # load original probabilities of model on dataset
        probs = self.load_probs(dataset, mode)

        # prepare output dictionary to gauge impact of word in prediction
        output = {}
        for ex_id in probs_filt:
            prob = probs[ex_id]
            prob_filt = probs_filt[ex_id]
            prob_diff = prob - prob_filt
            output[ex_id] = {'prob':float(prob[1]), 
                             'prob_filt':float(prob_filt[1]),
                             'prob_diff':float(prob_diff[1])}
        return output
    
    #== corpus word statistics analysis ===========================================================#
    def get_word_class_statistics(self, dataset, mode):
        data = self.data_handler.prep_split(dataset, mode)

        # go through data set and get word frequency
        num_labels = self.model_args.num_classes
        word_dict = defaultdict(lambda: np.zeros(num_labels))
        for ex in data:
            words = re.split('\W+', ex.text)
            for word in words:
                word_dict[word.lower()][ex.label] += 1
                
        return word_dict
    
