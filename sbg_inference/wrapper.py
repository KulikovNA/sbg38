import os
import sys
import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)

# Ensure SBG submodules are on PYTHONPATH
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

# `yolo_onnx.py` лежит рядом с этим файлом
from .yolo_onnx import YOLOv8Seg


class SBGGraspDetector:
    """Обёртка для инференса YOLOv8Seg + Scale‑Balanced‑Grasp (SBG).

    После инициализации вызывайте :py:meth:`infer`, чтобы получить:
      * ``gg_array``  – (N, 17) отфильтрованных захватов;
      * ``seg_vis``   – RGB‑кадр с цветными масками + центрами захватов;
      * ``mask_vis``  – RGB‑кадр только с цветными масками.
    """

    def __init__(
        self,
        checkpoint_path: str,
        onnx_seg: str,
        seg_yaml: str,
        *,
        num_view: int = 300,
        collision_thresh: float = 0.0,
        voxel_size: float = 0.01,
        bbox_depth_pad: float = 0.30,
        bbox_xy_pad: float = 0.00,
        max_grasp_num: int = 100,
        gripper_width_max: float = 1.2,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.bbox_depth_pad = bbox_depth_pad
        self.bbox_xy_pad = bbox_xy_pad
        self.max_grasp_num = max_grasp_num
        self.gripper_width_max = gripper_width_max
        self.conf_threshold = conf_threshold  
        self.iou_threshold = iou_threshold
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # SBG network
        self.net = GraspNet_MSCQ(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.08,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
            obs=False,
        ).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(state["model_state_dict"])
        self.net.eval()

        # Segmentation model
        self.seg_model = YOLOv8Seg(onnx_model=onnx_seg, yaml_path=seg_yaml)

    # ------------------------------------------------------------------
    def _mask_to_bbox3d(self, mask: np.ndarray, depth: np.ndarray, intr, depth_scale: float) -> Optional[np.ndarray]:
        """bool‑mask → (xmin,xmax,ymin,ymax,zmin,zmax) в системе камеры."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        z = depth[ys, xs].astype(np.float32) * depth_scale
        X = (xs - intr.ppx) / intr.fx * z
        Y = (ys - intr.ppy) / intr.fy * z
        ok = z > 0
        if not ok.any():
            return None
        X, Y, z = X[ok], Y[ok], z[ok]
        return np.array([
            X.min() - self.bbox_xy_pad,
            X.max() + self.bbox_xy_pad,
            Y.min() - self.bbox_xy_pad,
            Y.max() + self.bbox_xy_pad,
            z.min(),
            z.max() + self.bbox_depth_pad,
        ], np.float32)

    @staticmethod
    def _filter_centers_in_bbox(centers: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return (
            (centers[:, 0] >= xmin)
            & (centers[:, 0] <= xmax)
            & (centers[:, 1] >= ymin)
            & (centers[:, 1] <= ymax)
            & (centers[:, 2] >= zmin)
            & (centers[:, 2] <= zmax)
        )

    # ------------------------------------------------------------------
    def infer(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        intr,          # rs.intrinsics или namedtuple(fx, fy, ppx, ppy)
        depth_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вернёт (gg_array, seg_vis, mask_vis) — как в старом скрипте."""

        # 1. YOLO‑Seg → маски ------------------------------------------------
        boxes, segments, _ = self.seg_model(color_img, self.conf_threshold, self.iou_threshold)
        seg_vis = self.seg_model.draw_and_visualize(
                color_img.copy(), boxes, segments, vis=False, save=False)
        
        h, w = depth_img.shape
        masks: list[np.ndarray] = []
        for poly in segments:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            masks.append(m.astype(bool))

        mask_vis = seg_vis.copy()
        #rand_col = lambda: tuple(int(c) for c in np.random.randint(0, 255, 3))
        #for m in masks:
        #    mask_vis[m] = rand_col()
        

        # 2. Point cloud -----------------------------------------------------
        z = depth_img.astype(np.float32) * depth_scale
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        X = (xs - intr.ppx) / intr.fx * z
        Y = (ys - intr.ppy) / intr.fy * z
        valid = z > 0
        xyz = np.stack([X[valid], Y[valid], z[valid]], -1)
        if xyz.size == 0:
            return np.empty((0, 17), np.float32), seg_vis, mask_vis

        # 3. SBG inference ---------------------------------------------------
        idx = np.random.choice(xyz.shape[0], 20_000,
                            replace=xyz.shape[0] < 20_000)
        pc_t = torch.from_numpy(xyz[idx]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = pred_decode(self.net({"point_clouds": pc_t}))
        if not decoded:
            return np.empty((0, 17), np.float32), seg_vis, mask_vis

        gg_array = decoded[0].cpu().numpy()        # (k*17,) или (k,17)
        if gg_array.ndim == 1:                     # гарантируем (k,17)
            gg_array = gg_array.reshape(-1, 17)

        # 4. Collision check (по желанию) ------------------------------------
        if self.collision_thresh > 0 and gg_array.size:
            det  = ModelFreeCollisionDetector(xyz, voxel_size=self.voxel_size)
            mask = det.detect(GraspGroup(gg_array), 0.05, self.collision_thresh)
            gg_array = gg_array[~mask]

        # 5. BBox‑ и width‑фильтр -------------------------------------------
        if gg_array.size and masks:
            centers = gg_array[:, 13:16]
            inside  = np.zeros(len(centers), bool)
            for m in masks:
                bb = self._mask_to_bbox3d(m, depth_img, intr, depth_scale)
                if bb is not None:
                    inside |= self._filter_centers_in_bbox(centers, bb)
            inside &= gg_array[:, 1] <= self.gripper_width_max   # ≤ 60 мм
            gg_array = gg_array[inside]

        # 6. Отрисовка центров на seg_vis -----------------------------------
        if gg_array.size:
            centers = gg_array[:, 13:16]
            u = (centers[:, 0] * intr.fx / centers[:, 2] + intr.ppx).astype(int)
            v = (centers[:, 1] * intr.fy / centers[:, 2] + intr.ppy).astype(int)
            for ui, vi in zip(u, v):
                if 0 <= ui < w and 0 <= vi < h:
                    cv2.circle(seg_vis, (ui, vi), 5, (0, 255, 0), -1)

        # 7. Возвращаем «сырой» массив (k,17)  -------------------------------
        return gg_array, seg_vis, mask_vis



