import torch
import torch.nn.functional as F
import sacrebleu

from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss


class Seq2seqLoss(BaseLoss):
    def __init__(self, model, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        # Cross entropy loss
        loss = output.loss  

        # Token level accuracy
        mask = batch.label_ids != -100
        x = (output.logits.argmax(dim = -1) == batch.label_ids)
        hits = torch.masked_select(x, mask) 
        acc = hits.sum() / mask.sum()

        self.record_metrics({
            'loss': loss.item(),
            'acc':  acc.item(),
        })

        return SimpleNamespace(
                    loss=loss
        )

    def eval_corpus(self, batches) -> Tuple[float, dict]:
        pairs = {'ref': [], 'prd': []}
        NUM_BEAMS = 5

        for batch in batches:
            # Generate teacher forcing prediction
            output = self.model(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                labels = batch.label_ids,
            )

            # Token level accuracy
            mask = batch.label_ids != -100
            x = (output.logits.argmax(dim = -1) == batch.label_ids)
            hits = torch.masked_select(x, mask) 
            acc = hits.sum() / mask.sum()

            # Record accuracy scores
            self.record_metrics({
                'free_acc': acc,
            }, batch_size = mask.sum())

            # Generate free running prediction
            free_output = self.model.generate(
                input_ids = batch.input_ids, 
                attention_mask = batch.attention_mask, 
                max_length = 256,
                num_beams = NUM_BEAMS,
                length_penalty = 0.6,
                no_repeat_ngram_size = 4,
                num_return_sequences = 1,
                output_scores = True,
                return_dict_in_generate = True,
            )

            # Get the beam and decode
            beams = free_output.sequences
            texts = self.tokenizer.batch_decode(beams, skip_special_tokens=True)

            # Store results for corpus level computation
            pairs['ref'].extend(batch.label_text)
            pairs['prd'].extend(texts)


        # Corpus level scoring free-running
        freescore = sacrebleu.corpus_bleu(
            pairs['prd'],
            [pairs['ref']],
        ).score

        # Record accuracy scores
        self.record_metrics({
            'loss': -freescore,
            'bleu': freescore,
        }, batch_size = 1)

