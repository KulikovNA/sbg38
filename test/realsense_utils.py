# realsense_utils.py
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import os
import sys
import random

from config import cfgs

def get_realsense_pointcloud(pipeline, pc, align):
    """
    Снимок с камеры: получаем облако точек (xyz), цвета (rgb), intrinsics + сам color_image.
    Возвращаем кортеж (vtx, colors, (intr, color_image)).
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(float32).reshape(-1, 3)
    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(float32).reshape(-1, 2)

    h, w = color_image.shape[:2]
    u = (tex_coords[:, 0] * (w - 1)).astype(int)
    v = (tex_coords[:, 1] * (h - 1)).astype(int)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    colors = color_image[v, u, :]
    mask = np.isfinite(vtx).all(axis=1)
    vtx = vtx[mask]
    colors = colors[mask]

    # Если точек больше, чем cfgs.num_point — случайно отбираем
    if len(vtx) > cfgs.num_point:
        idx = np.random.choice(len(vtx), cfgs.num_point, replace=False)
        vtx = vtx[idx]
        colors = colors[idx]

    # Получаем intrinsics
    intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()

    return vtx, colors, (intr, color_image)


def preprocess_pointcloud(pointcloud, colors, num_points=20000):
    """
    Предобработка (фильтрация по NaN/inf и обрезка по диапазонам).
    Возвращаем (pointcloud_processed, colors_processed) или (None, None)
    если не хватило точек.
    """
    mask = ~np.isnan(pointcloud).any(axis=1) & \
           ~np.isinf(pointcloud).any(axis=1) & \
           np.any(pointcloud != 0, axis=1)
    pointcloud = pointcloud[mask]
    colors = colors[mask]

    mask_range = np.logical_and.reduce((
        pointcloud[:, 0] > -1.0,
        pointcloud[:, 0] < 1.0,
        pointcloud[:, 1] > -1.0,
        pointcloud[:, 1] < 1.0,
        pointcloud[:, 2] > 0.1,
        pointcloud[:, 2] < 2.0,
    ))
    pointcloud = pointcloud[mask_range]
    colors = colors[mask_range]

    if len(pointcloud) == 0:
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(float64) / 255.0)

    if len(pcd.points) >= num_points:
        ratio = num_points / len(pcd.points)
        pcd = pcd.random_down_sample(ratio)
    else:
        # Дополняем случайной выборкой
        needed = num_points - len(pcd.points)
        idx = np.random.choice(len(pcd.points), needed)
        pcd.points.extend(o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx]))
        pcd.colors.extend(o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx]))

    pointcloud_processed = np.asarray(pcd.points, dtype=float32)
    colors_processed = np.asarray(pcd.colors, dtype=float32)

    return pointcloud_processed, colors_processed
