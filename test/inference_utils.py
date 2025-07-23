# inference_utils.py

import torch
import numpy as np
import os
import sys

# Пути к корневой папке проекта и к нужным подкаталогам:
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI.grasp import GraspGroup
from collision_detector import ModelFreeCollisionDetector

from config import cfgs

def init_model():
    """
    Создаёт и инициализирует модель GraspNet_MSCQ, загружает чекпоинт, возвращает net в режиме eval().
    """
    net = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=cfgs.num_angle,
        num_depth=cfgs.num_depth,
        cylinder_radius=cfgs.cylinder_radius,
        hmin=cfgs.hmin,
        hmax_list=cfgs.hmax_list,
        is_training=False
    )
    net.to(cfgs.device)

    if os.path.isfile(cfgs.checkpoint_path):
        checkpoint = torch.load(cfgs.checkpoint_path, map_location=cfgs.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"-> Загружен чекпоинт (эпоха {checkpoint.get('epoch', '?')})")
    else:
        print(f"Файл чекпоинта не найден: {cfgs.checkpoint_path}")

    net.eval()
    return net


def run_inference(net, pointcloud):
    """
    Запускает инференс на переданном облаке точек (numpy, shape=(N,3)).
    Возвращает GraspGroup (все найденные захваты).
    """
    pc_torch = torch.from_numpy(pointcloud[np.newaxis, ...]).to(cfgs.device)

    with torch.no_grad():
        end_points = net({'point_clouds': pc_torch})
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    return gg


def filter_collisions(gg, pointcloud_processed, collision_detector, approach_dist=0.05):
    """
    Фильтрует захваты по коллизиям, используя ModelFreeCollisionDetector.
    Возвращает отфильтрованный GraspGroup.
    """
    if cfgs.collision_thresh > 0 and len(gg) > 0:
        # Если коллиз-детектор ещё не создан — создаём
        if collision_detector[0] is None:
            collision_detector[0] = ModelFreeCollisionDetector(
                pointcloud_processed,
                voxel_size=cfgs.voxel_size
            )
        mask = collision_detector[0].detect(
            gg,
            approach_dist=approach_dist,
            collision_thresh=cfgs.collision_thresh
        )
        gg = gg[~mask]
    return gg
