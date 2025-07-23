#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense + SBG + YOLOv8Seg

1.  Получаем кадр RealSense (RGB + depth) → строим облако точек.
2.  YOLOv8Seg выдаёт 2-D bbox-ы объектов.
3.  Для каждого bbox строим «грубый» 3-D параллелепипед, вытянутый от
    ближайшей глубины (z_near) до z_far = z_near + --bbox_depth_pad.
4.  Предсказания SBG (centers + approach) фильтруются:
      • центр должен попадать внутрь 3-D-bbox;
      • вектор approach направлен ≈ к камере (-approach·Z > cosθ).
5.  По каждому объекту печатаются лучшие N захватов.
6.  Визуализация Open3D остаётся **опциональной** (--no_vis).

Столбцы grasp-array (M,17):
  0-2  – центр       (x,y,z)   [м]
  3-5  – approach    (ax,ay,az)
  14   – score
  15   – width       [м]

Тест: Python 3.8, open3d 0.17, pyrealsense2 2.54, torch 2.2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch

# ────────────────────────────────────────────────
# project imports
# ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.extend([str(ROOT / "models"), str(ROOT / "utils")])

from graspnet import GraspNet_MSCQ, pred_decode           # noqa: E402
from graspnetAPI import GraspGroup                        # noqa: E402
from collision_detector import ModelFreeCollisionDetector # noqa: E402
from yolo_onnx import YOLOv8Seg                           # noqa: E402

# ────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────
def pixel2cam(u: np.ndarray, v: np.ndarray, z: np.ndarray, intr) -> Tuple[np.ndarray]:
    """(u,v,z) → (x,y,z)-в-камере."""
    X = (u - intr.ppx) / intr.fx * z
    Y = (v - intr.ppy) / intr.fy * z
    return X, Y, z


def bbox2d_to_bounds3d(
    bbox_xyxy: Tuple[int, int, int, int],
    depth: np.ndarray,
    depth_scale: float,
    intr,
    z_pad: float,
) -> Tuple[float, float, float, float, float, float]:
    """2-D bbox → (xmin,xmax,ymin,ymax,zmin,zmax) в координатах камеры."""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1] - 1, x2), min(depth.shape[0] - 1, y2)

    crop = depth[y1 : y2 + 1, x1 : x2 + 1]
    valid = crop > 0
    if valid.sum() == 0:
        return 0, 0, 0, 0, 1e9, -1e9  # «пустой» объём

    z_near = crop[valid].min() * depth_scale
    z_far = z_near + z_pad

    corners_px = np.array([(x1, y1), (x2, y1), (x1, y2), (x2, y2)], dtype=np.float32)
    u, v = corners_px[:, 0], corners_px[:, 1]

    x0, y0, _ = pixel2cam(u, v, np.full(4, z_near), intr)
    x1_, y1_, _ = pixel2cam(u, v, np.full(4, z_far), intr)

    xmin = min(x0.min(), x1_.min())
    xmax = max(x0.max(), x1_.max())
    ymin = min(y0.min(), y1_.min())
    ymax = max(y0.max(), y1_.max())

    return xmin, xmax, ymin, ymax, z_near, z_far


def filter_grasps_bbox(gg: np.ndarray, bounds) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    c = gg[:, :3]
    mask = (
        (c[:, 0] >= xmin)
        & (c[:, 0] <= xmax)
        & (c[:, 1] >= ymin)
        & (c[:, 1] <= ymax)
        & (c[:, 2] >= zmin)
        & (c[:, 2] <= zmax)
    )
    return gg[mask]


def filter_approach(gg: np.ndarray, cos_thresh: float = 0.5) -> np.ndarray:
    """Оставить захваты, подходящие «навстречу» камере (-Z вперёд)."""
    if gg.size == 0:
        return gg
    appr = gg[:, 3:6]
    appr_n = appr / (np.linalg.norm(appr, axis=1, keepdims=True) + 1e-8)
    dots = np.dot(appr_n, np.array([0, 0, 1.0]))  # cos between approach & +Z
    return gg[dots < -cos_thresh]


def sample_cloud(xyz: np.ndarray, n: int) -> np.ndarray:
    if xyz.shape[0] <= n:
        return xyz
    idx = np.random.choice(xyz.shape[0], n, replace=False)
    return xyz[idx]


# ────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SBG + YOLOv8Seg bbox-filter")
    p.add_argument("--checkpoint", required=True, help="checkpoint.tar SBG")
    p.add_argument("--max_points", type=int, default=20000, help="subsample N points")
    p.add_argument("--bbox_depth_pad", type=float, default=0.08, help="m")
    p.add_argument("--max_grasps", type=int, default=20)
    p.add_argument("--no_vis", action="store_true", help="не открывать окно Open3D")
    p.add_argument("--collision_thresh", type=float, default=0.0)
    p.add_argument("--voxel_size", type=float, default=0.01)
    p.add_argument("--orientation_cos", type=float, default=0.5, help="approach filter")
    return p.parse_args()


# ────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────
def main() -> None:
    args = get_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1 — SBG
    net = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.08,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"[INFO] SBG checkpoint loaded: {args.checkpoint}")

    # 2 — YOLOv8Seg
    seg = YOLOv8Seg(
        onnx_model="/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/best.onnx",
        yaml_path="/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/merged_yolo_dataset2.yaml",
    )

    # 3 — RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # опц-визуализация
    vis = None
    if not args.no_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window("RealSense + SBG", 1280, 720)

    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            d_fr, c_fr = frames.get_depth_frame(), frames.get_color_frame()
            if not d_fr or not c_fr:
                continue

            depth = np.asarray(d_fr.get_data())
            rgb = np.asarray(c_fr.get_data())

            # ── сегментация ───────────────────────────────
            bboxes, masks, _ = seg(rgb, conf_threshold=0.4, iou_threshold=0.5)

            # ── облако точек ─────────────────────────────
            h, w = depth.shape
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            z = depth.astype(np.float32) * depth_scale
            valid = z > 0
            X = (xs - intr.ppx) / intr.fx * z
            Y = (ys - intr.ppy) / intr.fy * z
            xyz_full = np.stack((X[valid], Y[valid], z[valid]), axis=-1)

            if xyz_full.shape[0] < 50:
                continue
            xyz_sample = sample_cloud(xyz_full, args.max_points)
            pc_t = torch.from_numpy(xyz_sample).float().unsqueeze(0).to(dev)

            # ── SBG ──────────────────────────────────────
            with torch.no_grad():
                gg_all = pred_decode(net({"point_clouds": pc_t}))[0].cpu().numpy()

            # ── обработка каждого bbox ───────────────────
            for idx, bbox in enumerate(bboxes):
                bounds = bbox2d_to_bounds3d(
                    bbox.astype(int), depth, depth_scale, intr, args.bbox_depth_pad
                )
                gg_obj = filter_grasps_bbox(gg_all, bounds)
                gg_obj = filter_approach(gg_obj, cos_thresh=args.orientation_cos)

                # коллизия (опц.)
                if args.collision_thresh > 0 and gg_obj.size > 0:
                    gg_tmp = GraspGroup(gg_obj)
                    mfc = ModelFreeCollisionDetector(xyz_full, voxel_size=args.voxel_size)
                    mask = mfc.detect(
                        gg_tmp,
                        approach_dist=0.05,
                        collision_thresh=args.collision_thresh,
                    )
                    gg_obj = gg_tmp[~mask].grasp_group_array

                if gg_obj.size == 0:
                    continue

                # sort & limit
                gg_obj = gg_obj[np.argsort(-gg_obj[:, 14])]  # по score
                gg_obj = gg_obj[: args.max_grasps]

                # ── вывод ─────────────────────────────────
                print(f"\nObject #{idx}  —  {gg_obj.shape[0]} grasps:")
                for g in gg_obj:
                    cx, cy, cz = g[:3]
                    score = g[14]
                    width = g[15]
                    print(
                        f"  center=({cx:+.3f},{cy:+.3f},{cz:+.3f}) "
                        f"score={score:+.3f}  w={width:.3f}"
                    )

                # ── визуализация (исключительно для grasp-ов) ──
                if not args.no_vis and gg_obj.size > 0:
                    vis.clear_geometries()
                    gg_vis = GraspGroup(gg_obj[:40])  # чуть меньше
                    for gr in gg_vis.to_open3d_geometry_list():
                        vis.add_geometry(gr)
                    vis.poll_events()
                    vis.update_renderer()

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        pipe.stop()
        if vis is not None:
            vis.destroy_window()


if __name__ == "__main__":
    main()
