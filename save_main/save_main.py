#save_main.py

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

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

"""
Суть работы скрипта:
- Загружает предварительно обученную модель SBG (GraspNet_MSCQ) для предсказания вариантов захвата (grasp) на основе облака точек.
- Инициализирует камеру Intel RealSense для получения цветовых и глубинных изображений.
- Преобразует глубинное изображение в облако точек с использованием внутренних параметров камеры.
- Сэмплирует облако точек и прогоняет его через нейронную сеть для получения предсказаний захвата (каждый захват описывается 17-ю числами).
- При необходимости выполняет детекцию коллизий для отсеивания невалидных вариантов захвата.
- Сохраняет облака точек (в формате PCD) и данные захватов (в формате JSON) для дальнейшего анализа.
- В режиме реального времени визуализирует облако точек и предсказанные захваты с использованием Open3D.
"""

def parse_args():
    parser = argparse.ArgumentParser("SBG RealSense Visualization + Saving")
    parser.add_argument('--checkpoint_path', required=True,
                        help='Путь к чекпоинту модели SBG')
    parser.add_argument('--num_view', type=int, default=300)
    parser.add_argument('--collision_thresh', type=float, default=0.0)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--num_point', type=int, default=20000)
    parser.add_argument('--obs', action='store_true')
    parser.add_argument('--max_grasp_num', type=int, default=100)
    parser.add_argument('--skip_frames', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='saved_data')
    return parser.parse_args()

def create_open3d_visualizer(window_name='RealSense + SBG', width=1280, height=720):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=width, height=height)
    return vis

def visualize_frame_and_grasps(vis, cloud_np, color_np, gg_array, max_grasp_num=50):
    vis.clear_geometries()

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud_np)
    pc.colors = o3d.utility.Vector3dVector(color_np)

    R = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])
    pc.rotate(R, center=[0, 0, 0])
    vis.add_geometry(pc)

    if gg_array is not None and gg_array.shape[0] > 0:
        gg_obj = GraspGroup(gg_array)
        gg_obj.nms()
        gg_obj.sort_by_score()
        if gg_obj.__len__() > max_grasp_num:
            gg_obj = gg_obj[:max_grasp_num]

        grippers = gg_obj.to_open3d_geometry_list()
        for g in grippers:
            g.rotate(R, center=[0, 0, 0])
            vis.add_geometry(g)

    vis.poll_events()
    vis.update_renderer()

def sample_point_cloud(xyz, color, num_point=20000):
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
    print("Параметры:", args)

    # 1) Модель
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
    print(f"[INFO] Загрузили модель SBG из {args.checkpoint_path}.")

    # 2) RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("[INFO] RealSense запущен.")

    for _ in range(args.skip_frames):
        pipeline.wait_for_frames()
    print(f"[INFO] Пропустили {args.skip_frames} кадров.\n")

    vis = create_open3d_visualizer("RealSense + SBG", 1280, 720)
    os.makedirs(args.save_dir, exist_ok=True)

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frame_count += 1

            depth_img = np.asanyarray(depth_frame.get_data())   # (480,640) uint16
            color_img = np.asanyarray(color_frame.get_data())   # (480,640,3) BGR

            # Формируем xyz
            z = depth_img.astype(float32) * depth_scale
            h, w = depth_img.shape
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            X = (xs - intr.ppx)/intr.fx * z
            Y = (ys - intr.ppy)/intr.fy * z

            valid = (depth_img > 0)
            xyz = np.stack([X[valid], Y[valid], z[valid]], axis=-1)  # (N,3)
            bgr = color_img[valid]                                   # (N,3) uint8

            # Пример порога по z
            mask_z = (xyz[:,2] < 1.2) & (xyz[:,2] > 0.2)
            xyz = xyz[mask_z]
            bgr = bgr[mask_z]
            if xyz.shape[0] < 5:
                visualize_frame_and_grasps(vis, np.empty((0,3)), np.empty((0,3)), None)
                continue

            color_f = bgr.astype(float32)/255.0
            xyz_s, color_s = sample_point_cloud(xyz, color_f, args.num_point)
            if xyz_s is None:
                visualize_frame_and_grasps(vis, np.empty((0,3)), np.empty((0,3)), None)
                continue

            # === !!! Явно приводим к float32 !!! ===
            xyz_s = xyz_s.astype(float32)  # ключевая строчка
            # =======================================
            pc_torch = torch.from_numpy(xyz_s).unsqueeze(0).to(device)  # (1, N, 3)

            # Прогоняем модель
            with torch.no_grad():
                end_points = {'point_clouds': pc_torch}
                out = net(end_points)
                grasp_preds = pred_decode(out)

            gg_array = grasp_preds[0].cpu().numpy()  # (M,17)
            if gg_array.shape[0] == 0:
                visualize_frame_and_grasps(vis, xyz_s, color_s, None, args.max_grasp_num)
                continue

            # Коллиз-детекция
            if args.collision_thresh > 0:
                gg_obj = GraspGroup(gg_array)
                mfcdetector = ModelFreeCollisionDetector(xyz_s, voxel_size=args.voxel_size)
                collision_mask = mfcdetector.detect(gg_obj,
                                                    approach_dist=0.05,
                                                    collision_thresh=args.collision_thresh)
                gg_obj = gg_obj[~collision_mask]
                gg_array = gg_obj.grasp_group_array
                if gg_array.shape[0] == 0:
                    visualize_frame_and_grasps(vis, xyz_s, color_s, None, args.max_grasp_num)
                    continue

            # Сохраняем результат
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_s)
            pcd.colors = o3d.utility.Vector3dVector(color_s)
            pcd_file = os.path.join(args.save_dir, f"cloud_{frame_count:06d}.pcd")
            o3d.io.write_point_cloud(pcd_file, pcd)

            json_file = os.path.join(args.save_dir, f"grasp_data_{frame_count:06d}.json")
            data_dict = {
                "frame_index": frame_count,
                "num_grasps": int(gg_array.shape[0]),
                "grasps_17": gg_array.tolist()
            }
            import json
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2)
            print(f"[SAVE] Кадр#{frame_count}: pcd => {pcd_file}, json => {json_file}")

            # Визуализация
            visualize_frame_and_grasps(vis, xyz_s, color_s, gg_array, args.max_grasp_num)

    except KeyboardInterrupt:
        print("[INFO] Выход по Ctrl+C")
    finally:
        pipeline.stop()
        vis.destroy_window()
        print("[INFO] Завершено.")


if __name__ == "__main__":
    main()
