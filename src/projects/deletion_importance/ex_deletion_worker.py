import re
import os
import random

from .ex_deteltion_search import ExampleDeletionAnalyser
from ...utils.general import save_json

BASE_PATH = '/home/al826/rds/hpc-work/2022/shortcuts/spurious-nlp/notebooks/deletion_importance/search_outputs'

class DeletionWorker:
    def __init__(self, path:str, device:str='cuda') -> None:
        self.system_path = path
        self.system = ExampleDeletionAnalyser(path, device)

    def deploy_worker(self, dataset:str, mode:str, metric:str='L2', maximise=False, rand=False):
        data = self.system.data_handler.prep_split(dataset, mode)
        output_path = self.get_output_path(dataset, mode, metric, maximise, rand)
        
        random.shuffle(data)

        for ex in data:
            ex_output_path = f"{output_path}/{ex.ex_id}"
            # if evaluation already done for ex, skip the example
            if os.path.isfile(ex_output_path):
                continue
            
            # create a temp file so another worker doesn't evaluate the ex
            print(ex.ex_id)
            save_json([None], ex_output_path)

            # run the evaluation loop
            ex_output = self.system.importance_search(ex, N=0, maximise=maximise, rand=rand)
            
            # save only specific keys
            keys = ['metrics_history', 'orginal_tokens', 'orginal_tokens_word', 'del_positions']
            ex_output_filt = {k: ex_output[k] for k in keys}

            save_json(ex_output_filt, ex_output_path)


    def get_output_path(self, dataset:str, mode:str, metric:str, maximise:bool, rand:bool):
        # get unique name for loaded model path
        model_name = re.sub(r'^.*?trained_models/', '', self.system_path)
        if model_name[-1] == '/': model_name = model_name[:-1]
        model_name = model_name.replace('/', '_')

        # get search type 
        search_name = f"{dataset}_{mode}"
        if rand:     search_name += '_rand'
        else:        search_name += f"_{metric}"
        if maximise: search_name += '_max'

        # put all together for overall output path
        output_path = f"{BASE_PATH}/{model_name}/{search_name}"

        # create model path if it does not already exist
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        return output_path