#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_image.py — инференс YOLOv8Seg + SBG для одной сохранённой пары color/depth.
Сохраняет:
  1. облако точек с захватами ДО фильтрации
  2. сегментированное изображение с проекцией захватов ДО фильтрации
  3. облако точек с захватами ПОСЛЕ фильтрации
  4. сегментированное изображение с проекцией захватов ПОСЛЕ фильтрации
"""

import os, sys, json, math, argparse
import numpy as np
import cv2
import torch
import open3d as o3d

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI import GraspGroup
from yolo_onnx import YOLOv8Seg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--color', default='data_rgbd/color_0000.png')
    p.add_argument('--depth', default='data_rgbd/depth_0000.png')
    p.add_argument('--intrinsics', default='data_rgbd/camera_intrinsics.json')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--iou_thresh', type=float, default=0.6)
    p.add_argument('--conf_thresh', type=float, default=0.5)
    p.add_argument('--num_view', type=int, default=300)
    p.add_argument('--collision_thresh', type=float, default=0.0)
    p.add_argument('--voxel_size', type=float, default=0.01)
    p.add_argument('--bbox_depth_pad', type=float, default=0.30)
    p.add_argument('--bbox_xy_pad', type=float, default=0.0)
    p.add_argument('--max_grasp_num', type=int, default=50)
    return p.parse_args()

def create_open3d_visualizer(window_name='Open3D', w=640, h=480, visible=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=w, height=h, visible=visible)
    return vis


def visualize_frame_and_grasps(vis, cloud_np, color_np, gg_array,
                               mask_visible=None):
    vis.clear_geometries()

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud_np)
    pc.colors = o3d.utility.Vector3dVector(color_np)
    R = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])
    pc.rotate(R, center=[0, 0, 0])
    vis.add_geometry(pc)

    if gg_array is not None and gg_array.shape[0]:
        if mask_visible is not None:
            gg_array = gg_array[mask_visible]
        gg_obj = GraspGroup(gg_array)
        for g in gg_obj.to_open3d_geometry_list():
            g.rotate(R, center=[0, 0, 0])
            vis.add_geometry(g)


def save_open3d_image(vis, path, wait_frames=10):
    for _ in range(wait_frames):
        vis.poll_events()
        vis.update_renderer()
    vis.capture_screen_image(path)
    print(f"[SAVED] Open3D image → {path}")


def load_intrinsics(path):
    with open(path, 'r') as f:
        return json.load(f)

def to_point_cloud(depth, intr):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) * intr['depth_scale']
    X = (xs - intr['ppx']) / intr['fx'] * z
    Y = (ys - intr['ppy']) / intr['fy'] * z
    valid = z > 0
    pts = np.stack([X[valid], Y[valid], z[valid]], axis=-1)
    return pts.astype(np.float32), valid

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.color))[0]

    intr = load_intrinsics(args.intrinsics)
    fx, fy, ppx, ppy = intr['fx'], intr['fy'], intr['ppx'], intr['ppy']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GraspNet_MSCQ(0, args.num_view, 12, 4, 0.08, -0.02, [0.01,0.02,0.03,0.04], False, False).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(state['model_state_dict']); net.eval()

    seg_model = YOLOv8Seg(
        onnx_model='/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/best.onnx',
        yaml_path ='/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/merged_yolo_dataset2.yaml')

    color_img = cv2.imread(args.color, cv2.IMREAD_COLOR)
    depth_img = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    boxes, segments, _ = seg_model(color_img, args.iou_thresh, args.conf_thresh)
    seg_vis = seg_model.draw_and_visualize(color_img.copy(), boxes, segments, vis=False, save=False)

    pts, valid_mask = to_point_cloud(depth_img, intr)
    colors = color_img[valid_mask] / 255.0
    if len(pts) > 20000:
        idx = np.random.choice(len(pts), 20000, replace=False)
        pts, colors = pts[idx], colors[idx]

    pc_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ep = net({'point_clouds': pc_t})
        gg_all = pred_decode(ep)[0].cpu().numpy()

    centers_all = gg_all[:, 13:16]
    u_all = (centers_all[:,0]*fx/centers_all[:,2] + ppx).astype(int)
    v_all = (centers_all[:,1]*fy/centers_all[:,2] + ppy).astype(int)
    h_img, w_img = color_img.shape[:2]

    mask_visible_all = (u_all >= 0) & (u_all < w_img) & (v_all >= 0) & (v_all < h_img)
    indices_visible = np.where(mask_visible_all)[0]

    max_vis = 60
    if len(indices_visible) > max_vis:
        selected_indices = np.random.choice(indices_visible, max_vis, replace=False)
    else:
        selected_indices = indices_visible

    gg_vis_before = gg_all[selected_indices]
    centers_vis_before = centers_all[selected_indices]

    seg_vis_before = seg_vis.copy()
    u_vis = (centers_vis_before[:, 0] * fx / centers_vis_before[:, 2] + ppx).astype(int)
    v_vis = (centers_vis_before[:, 1] * fy / centers_vis_before[:, 2] + ppy).astype(int)
    for ui, vi in zip(u_vis, v_vis):
        cv2.circle(seg_vis_before, (ui, vi), 5, (0, 255, 0), -1)

    vis_before = create_open3d_visualizer(window_name="cloud_before", visible=False)
    visualize_frame_and_grasps(vis_before, pts, colors, gg_vis_before)
    save_open3d_image(vis_before, os.path.join(args.output_dir, f"{base}_cloud_before.png"))
    vis_before.destroy_window()
    cv2.imwrite(os.path.join(args.output_dir, f"{base}_segm_before.png"), seg_vis_before)

    inside_all = np.zeros(len(gg_all), dtype=bool)

    for poly in segments:
        seg_mask = np.zeros(depth_img.shape, np.uint8)
        cv2.fillPoly(seg_mask, [poly.astype(np.int32)], 1)
        ys, xs = np.nonzero(seg_mask)
        z = depth_img[ys, xs].astype(np.float32) * intr['depth_scale']
        valid = z > 0
        if not np.any(valid): continue
        xs, ys, z = xs[valid], ys[valid], z[valid]
        X = (xs - ppx) / fx * z
        Y = (ys - ppy) / fy * z
        seg_pts = np.stack([X, Y, z], axis=-1)
        z_near = seg_pts[:, 2].min()
        z_far  = z_near + args.bbox_depth_pad
        xmin, xmax = seg_pts[:, 0].min(), seg_pts[:, 0].max()
        ymin, ymax = seg_pts[:, 1].min(), seg_pts[:, 1].max()
        inside_all |= (
            (centers_all[:, 0] >= xmin) & (centers_all[:, 0] <= xmax) &
            (centers_all[:, 1] >= ymin) & (centers_all[:, 1] <= ymax) &
            (centers_all[:, 2] >= z_near) & (centers_all[:, 2] <= z_far))

    final_mask = np.zeros_like(inside_all)
    for poly in segments:
        seg_mask = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(seg_mask, [poly.astype(np.int32)], 1)
        for idx in np.where(inside_all)[0]:
            u, v = u_all[idx], v_all[idx]
            if 0 <= u < w_img and 0 <= v < h_img:
                if seg_mask[v, u] > 0:
                    final_mask[idx] = True

    mask_after = final_mask[selected_indices]
    gg_vis_after = gg_vis_before[mask_after]
    centers_vis_after = centers_vis_before[mask_after]
    u_vis_after = (centers_vis_after[:,0]*fx/centers_vis_after[:,2] + ppx).astype(int)
    v_vis_after = (centers_vis_after[:,1]*fy/centers_vis_after[:,2] + ppy).astype(int)
    seg_vis_after = seg_model.draw_and_visualize(color_img.copy(), boxes, segments, vis=False, save=False)
    for ui, vi in zip(u_vis_after, v_vis_after):
        cv2.circle(seg_vis_after, (ui, vi), 5, (0,255,0), -1)

    if gg_vis_after.shape[0] > 0:
        vis_after = create_open3d_visualizer(window_name="cloud_after", visible=False)
        visualize_frame_and_grasps(vis_after, pts, colors, gg_vis_after)
        save_open3d_image(vis_after, os.path.join(args.output_dir, f"{base}_cloud_after.png"))
        vis_after.destroy_window()
    else:
        print("[INFO] No grasps left after filtering; skipping cloud_after.png.")

    cv2.imwrite(os.path.join(args.output_dir, f"{base}_segm_after.png"), seg_vis_after)
    print(f"[SAVED] RGB image → {base}_segm_after.png")
    print("Инференс завершён. Результаты сохранены в:", args.output_dir)


if __name__ == '__main__':
    main()
