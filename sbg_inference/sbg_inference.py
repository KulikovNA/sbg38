#sbg_inference.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import torch

# Если ваши файлы (graspnet.py, collision_detector.py, ...) лежат в ../models
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

"""
Суть работы скрипта:
- Загружает предварительно обученную модель SBG (GraspNet_MSCQ) для предсказания вариантов захвата (grasp) по облаку точек.
- Инициализирует камеру Intel RealSense для получения цветовых и глубинных изображений.
- Преобразует глубинное изображение в облако точек с использованием внутренних параметров камеры.
- Сэмплирует облако точек и пропускает его через нейронную сеть, получая предсказания захвата, описываемые 17-ю числами.
- При необходимости выполняет коллизионную проверку для отсеивания нежелательных вариантов захвата.
- В режиме реального времени визуализирует облако точек и предсказанные варианты захвата с использованием Open3D.
"""

def parse_args():
    parser = argparse.ArgumentParser("SBG RealSense Visualization")
    parser.add_argument('--checkpoint_path', required=True, help='Путь к чекпоинту модели SBG')
    parser.add_argument('--num_view', type=int, default=300, help='Число view для модели')
    parser.add_argument('--collision_thresh', type=float, default=0.0, help='Коллизионный порог (0 = нет проверки)')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Размер вокселя при коллиз. проверке')
    parser.add_argument('--num_point', type=int, default=20000, help='Максимум точек для семплирования')
    parser.add_argument('--obs', action='store_true', help='Включить ли obs-режим SBG (если нужен)')
    parser.add_argument('--max_grasp_num', type=int, default=100, help='Сколько захватов рисовать максимум')
    return parser.parse_args()

def create_open3d_visualizer(window_name='RealSense + SBG', width=640, height=480):
    """
    Создаём окно Open3D для визуализации.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=width, height=height)
    return vis

def visualize_frame_and_grasps(vis,
                               cloud_np,
                               color_np,
                               gg_array,
                               max_grasp_num=50):
    """
    Обновляет окно Open3D, отрисовывая облако точек и захваты.
    Параметры:
      - cloud_np: (N,3) float
      - color_np: (N,3) float, [0..1]
      - gg_array: (M,17) массив захватов
      - max_grasp_num: Сколько максимум захватов рисовать.
    """
    vis.clear_geometries()

    # Создаём облако точек
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud_np)
    pc.colors = o3d.utility.Vector3dVector(color_np)

    # Пример: поворачиваем облако на 180° вокруг оси X (можно убрать, если не нужно)
    R = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])
    pc.rotate(R, center=[0, 0, 0])
    vis.add_geometry(pc)

    if gg_array is not None and gg_array.shape[0] > 0:
        gg_obj = GraspGroup(gg_array)
        # По умолчанию в этой функции делается nms + сортировка по score, можно отключить
        gg_obj.nms()                # убираем близкие дубликаты
        gg_obj.sort_by_score()      # сортируем по убыванию score
        if gg_obj.__len__() > max_grasp_num:
            gg_obj = gg_obj[:max_grasp_num]

        grippers = gg_obj.to_open3d_geometry_list()
        for g in grippers:
            g.rotate(R, center=[0, 0, 0])  # такой же поворот
            vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

def sample_point_cloud(xyz, color, num_point=20000):
    """
    Семплировать облако до num_point точек (или дублировать, если меньше).
    """
    n = xyz.shape[0]
    if n == 0:
        return None, None
    if n >= num_point:
        idxs = np.random.choice(n, num_point, replace=False)
    else:
        idxs1 = np.arange(n)
        idxs2 = np.random.choice(n, num_point - n, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return xyz[idxs], color[idxs]

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Загружаем модель
    net = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=args.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.08,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
        obs=args.obs
    ).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f"Загрузили модель SBG из {args.checkpoint_path}.")

    # 2) Инициализация RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("RealSense запущен. Нажмите Ctrl+C для выхода...")

    # 3) Создаём окно Open3D
    vis = create_open3d_visualizer("RealSense + SBG", 1280, 720)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())   # (480,640), uint16
            color_img = np.asanyarray(color_frame.get_data())   # (480,640,3), BGR

            # Преобразуем depth + color => облако точек
            z = depth_img.astype(np.float32) * depth_scale
            h, w = depth_img.shape
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            X = (xs - intr.ppx) / intr.fx * z
            Y = (ys - intr.ppy) / intr.fy * z

            # валидные точки (где depth>0)
            valid = (depth_img > 0)
            xyz = np.stack([X[valid], Y[valid], z[valid]], axis=-1)
            # Цвет тоже берём
            color = color_img[valid]  # BGR uint8

            # Можно отбросить точки дальше 1.2м (пример)
            far_mask = (xyz[:,2] < 1.2) & (xyz[:,2] > 0.2)
            xyz = xyz[far_mask]
            color = color[far_mask]

            if xyz.shape[0] < 10:
                continue

            # Приведём color к float, [0..1]
            color_f = color.astype(np.float32) / 255.0


            # 4) Семплируем до num_point
            xyz_s, color_s = sample_point_cloud(xyz, color_f, num_point=args.num_point)
            if xyz_s is None:
                continue

            # 5) Запускаем модель
            pc_torch = torch.from_numpy(xyz_s).float().unsqueeze(0).to(device)  # (1, N, 3)
            end_points = {'point_clouds': pc_torch}
            with torch.no_grad():
                out = net(end_points)
                grasp_preds = pred_decode(out)  # list of length B=1 => grasp_preds[0] shape: (M,17)

            gg_array = grasp_preds[0].cpu().numpy()  # (M,17)

            # 6) Коллизионная проверка (необязательна)
            if args.collision_thresh > 0 and gg_array.shape[0] > 0:
                gg_obj = GraspGroup(gg_array)
                mfcdetector = ModelFreeCollisionDetector(xyz, voxel_size=args.voxel_size)
                collision_mask = mfcdetector.detect(gg_obj,
                                                    approach_dist=0.05,
                                                    collision_thresh=args.collision_thresh)
                gg_obj = gg_obj[~collision_mask]
                gg_array = gg_obj.grasp_group_array

            # 7) Визуализация
            visualize_frame_and_grasps(vis,
                                       xyz,      # (N,3), без поворота здесь
                                       color_f,  # (N,3), [0..1]
                                       gg_array, # (M,17)
                                       max_grasp_num=args.max_grasp_num)
    except KeyboardInterrupt:
        print("Выход по Ctrl+C")
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()
