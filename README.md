# Input Filtering

### Dependencies
You'll need to install transformers, torch, wandb, scipy, nltk

### Training classification Model
Example of training script

```
python run_train.py --path trained_models/EXP-NAME/seed-1 --dataset imdb --filters NV --transformer bert-base
```

- ```--path``` sets the experiment directory. Extra seeds can be trained by simply using ```EXP-NAME/seed-k``` instead of ```EXP-NAME/seed-1``` (seeds are reproducible)
- ```--dataset``` sets the data to train the model on. classification datasets must be interfaced in ```src/data/handler.py```
- ```--filters``` sets the filtering process for the inputs. ```NV``` relates to only keeping nouns and verbs- other options can be found in ```src/data/word_filter.py```
- ```--transformer``` sets the base pre-trained transformer to initialise model weights with. interfaced transformers can be found in ```src/models/pre_trained_trans.py```
- other training arguments can be used/modified, look into train.py to see which arguments are available.

### Training sequence to sequence Model
```
python run_train.py --path trained_models/EXP-NAME/seed-1 --data-set wmt16 --filters NV --transformer t5-base
```

has similar arguments, though with a seq2seq dataset and encoder-decoder transformer model, then model will be trained for seq2seq.
