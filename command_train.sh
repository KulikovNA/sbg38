#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py  \
    --log_dir logs/log_new_model  \
    --camera realsense \
    --batch_size 2 \
    --dataset_root /mnt/sda1/Scale-Balanced-Grasp/data/mahaoxiang/graspnet

