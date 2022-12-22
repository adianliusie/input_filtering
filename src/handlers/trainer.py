import os
import logging
import wandb
import torch
import re

from collections import namedtuple
from types import SimpleNamespace
from typing import Optional
from tqdm import tqdm


from .batcher import Batcher, Seq2seqBatcher
from ..data.handler import DataHandler
from ..models.models import TransformerModel 
from ..models.pre_trained_trans import load_seq2seq_transformer, SEQ2SEQ_MODELS
from ..utils.general import save_json, load_json
from ..utils.torch import set_rand_seed
from ..loss.cross_entropy import CrossEntropyLoss
from ..loss.seq2seq import Seq2seqLoss

# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(object):
    """ Base class for finetuning transformer to datasets """
    def __init__(self, path: str, args: namedtuple):
        self.setup_exp(path, args)
        self.setup_helpers(args)

    def setup_helpers(self, args: namedtuple):
        # set random seed
        if (args.rand_seed is None) and ('/seed-' in self.exp_path):
            args.rand_seed = int(self.exp_path.split('/seed-')[-1])
        set_rand_seed(args.rand_seed)

        # set up attributes 
        self.model_args = args
        self.data_handler = DataHandler(trans_name=args.transformer, filters=args.filters)

        if args.transformer in SEQ2SEQ_MODELS:
            self.batcher = Seq2seqBatcher(max_len=args.maxlen)
            self.model = load_seq2seq_transformer(args.transformer)
            self.model_loss = Seq2seqLoss(self.model, self.data_handler.tokenizer)

        else:
            self.batcher = Batcher(max_len=args.maxlen)
            self.model = TransformerModel(trans_name=args.transformer, num_classes=args.num_classes)
            self.model_loss = CrossEntropyLoss(self.model)

    #== Main Training Methods =====================================================================#
    def train(self, args: namedtuple):
        self.save_args('train-args.json', args)
 
        # set up optimization objects
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr)

        # set up model
        self.to(args.device)
        self.model.train()
        self.log_num_params()

        # Reset loss metrics
        self.best_dev = (0, {})
        self.model_loss.reset_metrics()

        # Get train, val, test split of data
        train, dev, test = self.data_handler.prep_data(args.dataset, args.lim)

        # Setup wandb for online tracking of experiments
        if args.wandb: self.setup_wandb(args)

        for epoch in range(1, args.epochs+1):
            # freeze transformer for first couple of epochs if set
            if args.freeze_trans:
                if epoch <= args.freeze_trans: self.model.freeze_transformer()
                else: self.model.unfreeze_transformer()

            #== Training =============================================
            train_batches = self.batcher(
                data = train, 
                bsz = args.bsz, 
                shuffle = True
            )
            for step, batch in enumerate(train_batches, start = 1):
                output = self.model_loss(batch)

                optimizer.zero_grad()
                output.loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                optimizer.step()
        
                # Print train performance every log_every samples
                if step % (args.log_every//args.bsz) == 0:
                    metrics = self.get_metrics()
                    
                    self.log_metrics(
                        metrics = metrics,
                        mode = 'train', 
                        epoch = epoch,
                        ex_step = step*args.bsz
                    )

                    if args.wandb: self.log_wandb(metrics, mode='train')
                    self.model_loss.reset_metrics()   

                # if val_every given then run validation within epoch
                if step % (args.val_every//args.bsz) == 0:
                    self.validate(dev, epoch, step*args.bsz, args.wandb)

            #== Validation ============================================
            self.validate(dev, epoch, ex_step='end', wandb=args.wandb)

            best_epoch = int(self.best_dev[0].split('-')[0])
            if epoch - best_epoch >= 3:
                break

    def validate(self, dev, epoch:int, ex_step:int=None, wandb=False):
        metrics = self.run_validation(dev, mode='dev')
        self.log_metrics(metrics = metrics, mode='dev')
        if wandb: self.log_wandb(metrics, mode= 'dev')

        if metrics['loss'] < self.best_dev[1].get('loss', float('inf')):
            self.best_dev = (f'{epoch}-{ex_step}', metrics.copy())
            self.save_model()
        
        self.log_metrics(metrics=self.best_dev[1], mode='dev-best', ex_step=self.best_dev[0])

    @torch.no_grad()
    def run_validation(self, data, bsz:int=1, mode='dev'):
        self.model_loss.reset_metrics()

        val_batches = self.batcher(
            data = data, 
            bsz = bsz, 
            shuffle = False
        )

        for batch in val_batches:
            self.model_loss.eval_forward(batch)
        
        metrics = self.get_metrics()
        return metrics

    #== Logging Utils =============================================================================#
    def get_metrics(self):
        metrics = {key: value.avg for key, value in self.model_loss.metrics.items()}
        return metrics

    def log_metrics(self, metrics: dict, mode: str, epoch:str=None, ex_step:int=None):
        # Create logging header
        if   mode == 'train'        : msg = f'epoch {epoch:<2}   ex {ex_step:<7} '
        elif mode in ['dev', 'test']: msg = f'{mode:<10}' + 12*' '
        elif mode == 'dev-best'     : msg = f'best-dev {"(" + ex_step + ")":<12} '    
        else: raise ValueError()

        # Get values from Meter and print all
        for key, value in metrics.items():
            msg += f'{key}: {value:.3f}  '
        
        # Log Performance 
        logger.info(msg)

    def log_wandb(self, metrics, mode):
        if mode != 'train': 
            metrics = {f'{mode}-{key}': value for key, value in metrics.items()}
        wandb.log(metrics)

    #== Saving Utils ==============================================================================#
    def save_args(self, name: str, data: namedtuple):
        """ Saves arguments into json format """
        path = os.path.join(self.exp_path, name)
        save_json(data.__dict__, path)

    def load_args(self, name: str) -> SimpleNamespace:
        path = os.path.join(self.exp_path, name)
        args = load_json(path)
        return SimpleNamespace(**args)
    
    def save_model(self, name : str ='model'):
        # Get current model device
        device = next(self.model.parameters()).device
        
        # Save model in cpu
        self.model.to("cpu")
        path = os.path.join(self.exp_path, 'models', f'{name}.pt')
        torch.save(self.model.state_dict(), path)

        # Return to original device
        self.model.to(device)

    def load_model(self, name: str = 'model'):
        name = name if name is not None else 'model'
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.exp_path, 'models', f'{name}.pt')
            )
        )

    #== Experiment Utils ==========================================================================#
    def setup_exp(self, exp_path: str, args: namedtuple):
        self.exp_path = exp_path

        # prepare experiment directory structure
        if not os.path.isdir(self.exp_path):
            os.makedirs(self.exp_path)

        mod_path = os.path.join(self.exp_path, 'models')
        if not os.path.isdir(mod_path):
            os.makedirs(mod_path)

        eval_path = os.path.join(self.exp_path, 'eval')
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)

        # add file handler to logging
        fh = logging.FileHandler(os.path.join(exp_path, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        #save model arguments
        self.save_args('model_args.json', args)

    def to(self, device):
        assert hasattr(self, 'model') and hasattr(self, 'batcher')
        self.model.to(device)
        self.batcher.to(device)

    def setup_wandb(self, args: namedtuple):
        # remove everything before */trained_models for exp_name
        exp_name = re.sub(r'^.*?trained_models', '', self.exp_path)

        # remove the final -seed-i from the group name
        group_name = '/seed'.join(exp_name.split('/seed')[:-1])

        #init wandb project
        wandb.init(
            project='shortcuts-{}'.format(args.dataset),
            entity='adian',
            name=exp_name, 
            group=group_name,
            dir=self.exp_path
        )

        # save experiment config details
        cfg = {
            'dataset': args.dataset,
            'bsz': args.bsz,
            'lr': args.lr,
            'transformer': self.model_args.transformer,
        }

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    def log_num_params(self):
        """ prints number of paramers in model """
        logger.info("Number of parameters in model {:.1f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6
        ))


