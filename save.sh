#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python save_main/save_main.py \
    --checkpoint_path logs/log_full_model/checkpoint.tar 
