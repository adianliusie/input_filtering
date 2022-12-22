from transformers import ElectraModel, BertModel, BertConfig, RobertaModel, AutoModel
from transformers import T5ForConditionalGeneration

def load_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained transformer from huggingface """
    if   system == 'bert-base'    : trans_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system == 'bert-rand'    : trans_model = BertModel(BertConfig())
    elif system == 'bert-large'   : trans_model = BertModel.from_pretrained('bert-large-uncased', return_dict=True)
    elif system == 'bert-tiny'    : trans_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    elif system == 'roberta-base' : trans_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system == 'electra-base' : trans_model = ElectraModel.from_pretrained('google/electra-base-discriminator',return_dict=True)
    elif system == 'electra-large': trans_model = ElectraModel.from_pretrained('google/electra-large-discriminator', return_dict=True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model

SEQ2SEQ_MODELS = ['t5-small', 't5-base', 't5-large']
def load_seq2seq_transformer(system:str)->'Model':
    if   system == 't5-small' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    elif system == 't5-base'  : trans_model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    elif system == 't5-large' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-large", return_dict = True)
    else: raise ValueError(f"invalid transfomer system provided: {system}")
    return trans_model