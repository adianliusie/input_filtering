import torch
import torch.nn.functional as F
import random

from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict

from ...handlers.evaluater import Evaluator

class ExampleDeletionAnalyser(Evaluator):
    def __init__(self, path, device='cuda'):
        # load project
        self.exp_path = path
        super().setup_helpers()
        self.model.eval()

        #set device
        self.to(device)
        self.device = device

    @torch.no_grad()
    def ex_importance_search(self, ex, metric='L2', N=None, eps=None, maximise=False, rand=False, quiet=False):
        """ given an example, finds the words that can be deleted with minimal influence on the prediction """
        #only one of N or eps is given, and that maximise and random are both not true
        assert bool(N is None) != bool(eps is None)
        assert not (maximise and rand)

        # get original model output
        batch = self.tokenize_ex(ex)
        output = self.model_loss(batch)
        
        # variables for the search
        del_positions = []        
        attn_mask = deepcopy(batch.attention_mask)
        seq_len = batch.input_ids.size(-1)
        cur_dist = 0

        # init variables
        condition = True
        metrics_history = []

        # do the brute search
        while condition:
            num_tokens = int(attn_mask.sum().item()) - 1
            position_scores = defaultdict(lambda x: 0)

            #stop if only CLS and SEP are left
            if num_tokens <= 2: break

            for pos in range(1, seq_len-1):
                # skip positions already deleted
                if pos in del_positions: continue
                
                # prepare by masking current position
                mask = attn_mask.clone()
                mask[:,pos] = 0
                batch.attention_mask = mask

                # get score after masking current position
                del_output = self.model_loss(batch)
                position_scores[pos] = self.dist_fn(output, del_output, batch)

            # select the position that cuases smallest change in metric
            if maximise:
                k_best, dist_dict = max(position_scores.items(), key=lambda x: x[1][metric])
            elif rand:
                k_best, dist_dict = random.choice(list(position_scores.items()))
            else:
                k_best, dist_dict = min(position_scores.items(), key=lambda x: x[1][metric])
            cur_dist = dist_dict[metric]
            
            # print current state
            msg = f"{num_tokens:<3} {k_best:<3} "
            for key, value in dist_dict.items():
                if key in ['L1', 'L2', 'cosine', 'prob', 'pred', 'log_prob_diff']:
                    msg += f'{key}: {value:.4f}  '
            if not quiet: print(msg)

            #log distances for tracking
            metrics_history.append(dist_dict)

            # check if condition is still satisfied
            if N is not None:
                condition = num_tokens > N
            elif eps is not None:
                condition = (cur_dist < eps) and num_tokens > 0

            # update attention mask if search continues
            if condition:
                attn_mask[:,k_best] = 0
                del_positions.append(k_best)

        # return output of the search
        orginal_tokens_word = [self.data_handler.tokenizer.decode(k) for k in batch.input_ids[0]]
        original_text = self.data_handler.tokenizer.decode(batch.input_ids[0])
        final_input_ids = batch.input_ids[attn_mask.bool()]
        final_text = self.data_handler.tokenizer.decode(final_input_ids)

        return {'original_text': original_text, 
                'final_text':final_text, 
                'orginal_tokens':batch.input_ids[0].tolist(),
                'final_tokens':final_input_ids.tolist(),
                'orginal_tokens_word':orginal_tokens_word,
                'del_positions': del_positions,
                'metrics_history':metrics_history
            }

    #== General util methods ======================================================================#
    def load_ex(self, dataset:str, mode:str, k:int=0)->SimpleNamespace:
        data = self.data_handler.prep_split(dataset, mode, lim=10) #TEMP
        ex = data[k]
        return ex

    def tokenize_ex(self, ex:SimpleNamespace):
        if self.model_args.num_classes == 3: 
            ex = self.data_handler._prep_ids_pairs([ex])
        else: 
            ex = self.data_handler._prep_ids([ex])
        batch = next(self.batcher(ex, bsz=1))
        return batch

    def dist_fn(self, output_og:SimpleNamespace, output:SimpleNamespace, batch:SimpleNamespace)->dict:
        """ calculates various distances between the 2 outputs"""
        # get logits and hidden representations
        h_og, h = output_og.h[0], output.h[0]
        logits_og, logits = output_og.logits[0], output.logits[0]

        # get predictions and probability distributions of prediction
        pred = int(torch.argmax(logits, axis=-1).item())
        prob_distr = F.softmax(logits, dim=-1)
        
        # get the probability of the original prediction (i.e. input wihtout deletion)
        pred_og = int(torch.argmax(logits_og, axis=-1).item())
        prob = prob_distr[pred_og].item()

        # get probability of the label class
        label = int(batch.labels.item())
        label_prob = prob_distr[label].item()

        # distance metrics
        L1 = torch.mean(torch.abs(h-h_og)).item()
        L2 = torch.sqrt(torch.mean((h-h_og)**2)).item()
        cosine = torch.sum(h * h_og)/torch.sqrt((h*h).sum() * (h_og*h_og).sum())
        log_prob_diff = (logits_og[pred_og]-logits[pred_og]).item()

        return {'prob_dist':prob_distr.tolist(),
                'prob':float(prob),
                'label_prob':float(label_prob),
                'pred':int(pred),
                'label':int(label),
                'log_prob_diff':float(log_prob_diff),
                'L1':float(L1), 
                'L2':float(L2),
                'cosine':float(cosine),
                }

