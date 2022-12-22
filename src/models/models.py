import torch
import torch.nn as nn

from types import SimpleNamespace
from transformers import logging

from .pre_trained_trans import load_transformer

logging.set_verbosity_error()

class TransformerModel(torch.nn.Module):
    """basic transformer model for multi-class classification""" 
    def __init__(self, trans_name:str, num_classes:int=2):
        super().__init__()
        self.transformer = load_transformer(trans_name)
        h_size = self.transformer.config.hidden_size
        self.output_head = nn.Linear(h_size, num_classes)
        
    def forward(self, *args, **kwargs):
        trans_output = self.transformer(*args, **kwargs)
        H = trans_output.last_hidden_state  #[bsz, L, 768] 
        h = H[:, 0]                         #[bsz, 768] 
        logits = self.output_head(h)             #[bsz, C] 
        return SimpleNamespace(h=h, logits=logits)

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
    
    def freeze_classifier_bias(self):
        self.output_head.bias.requires_grad = False