from datasets import load_dataset
from typing import List, Dict, Tuple, TypedDict
from tqdm import tqdm

from .load_classification_hf import _rename_keys

class Seq2SeqText(TypedDict):
    """Output example formatting (only here for documentation)"""
    input_text : str
    ref_text : str

#== Main loading function =========================================================================# 

HF_SEQ2SEQ_DATA = ['wmt16', 'cnn-dm']
def load_hf_seq2seq_data(data_name):
    """ loading NLI datsets available on huggingface hub """
    if   data_name == 'wmt16' : train, dev, test = load_wmt16()
    elif data_name == 'cnn-dm': train, dev, test = load_cnn_dailymail()
    else: raise ValueError(f"invalid text pair dataset name: {data_name}")
    return train, dev, test
            
#== seq2seq data set loader =======================================================================#
def format_wmt16(data: List[Dict], lang: str) -> List[Dict]:
    """ Converts data split on huggingface into seq2seq format for the framework """
    output = []
    for ex in tqdm(data):
        ex = ex['translation']
        output.append({
            'input_text': ex['en'], 
            'label_text': ex[lang],
        })
    return output

def load_wmt16(lang:str='de'):
    """ Loads wmt16-de-en data from huggingface """
    data = load_dataset("wmt16", "{}-en".format(lang))
    train, dev, test = data['train'], data['validation'], data['test']
    train, dev, test = [format_wmt16(split, lang) for split in [train, dev, test]]
    return train, dev, test

def load_cnn_dailymail():
    data = load_dataset("ccdv/cnn_dailymail", '3.0.0')
    train = list(data['train'])
    dev   = list(data['validation'])
    test  = list(data['test'])

    train, dev, test = _rename_keys(train, dev, test, old_key='article', new_key='input_text')
    train, dev, test = _rename_keys(train, dev, test, old_key='highlights', new_key='label_text')
    return train, dev, test
