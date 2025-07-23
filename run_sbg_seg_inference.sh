#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python sbg_inference/sbg_inference_seg.py \
    --checkpoint_path logs/log_full_model/checkpoint.tar \
    --collision_thresh 0 \