# config.py
import torch
import os

class Config:
    num_point = 50000
    num_view = 300
    checkpoint_path = 'logs/log_full_model/checkpoint.tar'
    dump_dir = 'logs/dump_full_mode'
    collision_thresh = 0.01
    voxel_size = 0.01
    num_angle = 12
    num_depth = 4
    cylinder_radius = 0.08
    hmin = -0.02
    hmax_list = [0.01, 0.02, 0.03, 0.04]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Создадим экземпляр для удобства
cfgs = Config()
