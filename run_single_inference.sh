#!/bin/bash

# выбирать нужный GPU
export CUDA_VISIBLE_DEVICES=0

# пути по умолчанию (меняйте под себя)
COLOR_IMG="/home/nikita/diplom/Scale-Balanced-Grasp/data_rgbd/color_0005.png"
DEPTH_IMG="/home/nikita/diplom/Scale-Balanced-Grasp/data_rgbd/depth_0005.png"
INTRINSICS="/home/nikita/diplom/Scale-Balanced-Grasp/data_rgbd/camera_intrinsics.json"
CHECKPOINT="logs/log_full_model/checkpoint.tar"
OUTPUT_DIR="output"

# создаём папку для результатов
mkdir -p "${OUTPUT_DIR}"

# прогон инференса
python sbg_inference/inference_single.py \
    --checkpoint_path "${CHECKPOINT}" \
    --color         "${COLOR_IMG}" \
    --depth         "${DEPTH_IMG}" \
    --intrinsics    "${INTRINSICS}" \
    --output_dir    "${OUTPUT_DIR}" \
    --iou_thresh    0.6 \
    --conf_thresh   0.5 \
    --collision_thresh 0 \
    --num_view      300 \
    --bbox_depth_pad 0.30 \
    --bbox_xy_pad   0.0 \
    --max_grasp_num 50
