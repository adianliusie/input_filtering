#!/bin/bash

# activate environment
# source /home/al826/rds/hpc-work/envs/env_1/bin/activate
source /home/al826/rds/rds-altaslp-8YSp2LXTlkY/experiments/yf286/venv/ssl/bin/activate

#load any enviornment variables needed
source ~/.bashrc

TOKENIZERS_PARALLELISM=false

python ../eval.py $@
