#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RealSense + YOLOv8Seg + Scale-Balanced-Grasp (SBG).

* Оси камеры:  X — вправо, Y — вниз, Z — вперёд.
* Строим 3-D-bbox по маске сегментации + глубине (с паддингом по X,Y и Z).
* Оставляем только те захваты, чьи центры попали в хотя бы один 3-D-bbox.
"""

import os, sys, math, argparse
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import torch, cv2
import copy
# ====== SBG (GraspNet-MSCQ) ======
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

# ====== YOLOv8 Segmentation ======
from yolo_onnx import YOLOv8Seg

import socket

HOST, PORT = "127.0.0.1", 6000

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("RealSense → SBG → YOLO mask filter")
    p.add_argument('--checkpoint_path', required=True, help='Чекпойнт SBG')
    p.add_argument('--num_view', type=int, default=300)
    p.add_argument('--collision_thresh', type=float, default=0.0)
    p.add_argument('--voxel_size', type=float, default=0.01)
    p.add_argument('--bbox_depth_pad', type=float, default=0.30,
                   help='Сколько добавить к z-размеру bbox (м)')
    p.add_argument('--bbox_xy_pad', type=float, default=0.0,
                   help='Паддинг по X,Y (м) для bbox и фильтра')
    p.add_argument('--max_grasp_num', type=int, default=50)
    p.add_argument('--show_axes',  action='store_true')
    p.add_argument('--show_bbox',  action='store_true')
    p.add_argument('--show_inliers', action='store_true',
                   help='Показывать только точки, попавшие в 3-D-bbox')
    return p.parse_args()


# ----------------------------------------------------------------------
#  Open3D helpers
# ----------------------------------------------------------------------
def aabb_to_lineset(bbox, color=(0, 1, 0)):
    """bbox = (xmin,xmax,ymin,ymax,zmin,zmax) → open3d.geometry.LineSet"""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    pts = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax]
    ], dtype=np.float32)
    lines = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ], dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines =o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, np.float32), (lines.shape[0],1)))
    return ls


def create_open3d_visualizer(name='SBG viewer', w=640, h=480):
    vis = o3d.visualization.Visualizer()
    vis.create_window(name, width=w, height=h)
    return vis


def visualize_frame_and_grasps(
        vis, cloud_np, color_np,
        gg_array, bbox_list=None,
        show_axes=False, show_bbox=False,
        max_grasp_num=50):
    """Полная сцена — облако, захваты, bbox-ы, оси."""
    vis.clear_geometries()

    # point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud_np)
    pc.colors = o3d.utility.Vector3dVector(color_np)
    R = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])
    pc.rotate(R, center=[0, 0, 0]);  vis.add_geometry(pc)

    if show_axes:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        cf.rotate(R, center=[0, 0, 0]);  vis.add_geometry(cf)

    if show_bbox and bbox_list:
        for bb in bbox_list:
            ls = aabb_to_lineset(bb)
            ls.rotate(R, center=[0, 0, 0]);  vis.add_geometry(ls)

    if gg_array is not None and gg_array.shape[0]:
        gg_obj = GraspGroup(gg_array)
        gg_obj.nms();  gg_obj.sort_by_score()
        if len(gg_obj) > max_grasp_num:
            gg_obj = gg_obj[:max_grasp_num]
        for g in gg_obj.to_open3d_geometry_list():
            g.rotate(R, center=[0, 0, 0]);  vis.add_geometry(g)

    vis.poll_events();  vis.update_renderer()


# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------
def sample_point_cloud(xyz, color, num_point=2000):
    if xyz.shape[0] == 0:
        return None, None
    idx = np.random.choice(xyz.shape[0], num_point,
                           replace=xyz.shape[0] < num_point)
    return xyz[idx], color[idx]


def mask_to_bbox3d(mask, depth_img, intr, depth_scale,
                   depth_pad, xy_pad):
    """bool-mask → (xmin,xmax,ymin,ymax,zmin,zmax) в камере."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    z = depth_img[ys, xs].astype(np.float32) * depth_scale
    X = (xs - intr.ppx) / intr.fx * z
    Y = (ys - intr.ppy) / intr.fy * z
    valid = z > 0
    if not valid.any():
        return None
    X, Y, z = X[valid], Y[valid], z[valid]
    xmin, xmax = X.min() - xy_pad, X.max() + xy_pad
    ymin, ymax = Y.min() - xy_pad, Y.max() + xy_pad
    zmin       = z.min()
    zmax       = z.max() + depth_pad      # ←-- исправлено
    return np.array([xmin, xmax, ymin, ymax, zmin, zmax], np.float32)


def filter_grasps_in_bbox(centers, bbox):
    xmin,xmax,ymin,ymax,zmin,zmax = bbox
    return ((centers[:,0]>=xmin)&(centers[:,0]<=xmax)&
            (centers[:,1]>=ymin)&(centers[:,1]<=ymax)&
            (centers[:,2]>=zmin)&(centers[:,2]<=zmax))


def filter_points_in_bbox(xyz, bbox):
    xmin,xmax,ymin,ymax,zmin,zmax = bbox
    return ((xyz[:,0]>=xmin)&(xyz[:,0]<=xmax)&
            (xyz[:,1]>=ymin)&(xyz[:,1]<=ymax)&
            (xyz[:,2]>=zmin)&(xyz[:,2]<=zmax))


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m21 - m12) * s
        qy = (m02 - m20) * s
        qz = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)

def format_pose_line(row: np.ndarray) -> str:
    t = row[13:16]
    R = row[4:13].reshape(3, 3)
    q = rotation_matrix_to_quaternion(R)
    vals = np.concatenate([t, q])
    return " ".join(f"{v:.6f}" for v in vals) + "\n"

def send_poses(gg_array: np.ndarray):
    # инициализация сервера (как раньше)
    if not hasattr(send_poses, "server"):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        srv.setblocking(False)
        send_poses.server = srv
        print(f"[POSE-SERVER] listening on {HOST}:{PORT}")

    # если нет захватов — ничего не отправляем
    if gg_array.shape[0] == 0:
        return

    # формируем все строки с позами
    lines = [format_pose_line(r) for r in gg_array]

    # инициализируем счётчик, если ещё нет
    if not hasattr(send_poses, "pose_idx"):
        send_poses.pose_idx = 0

    # выбираем одну строку по модулю длины списка
    idx = send_poses.pose_idx % len(lines)
    line = lines[idx]

    # увеличиваем счётчик для следующей отправки
    send_poses.pose_idx += 1

    # пытаемся принять подключение
    try:
        conn, addr = send_poses.server.accept()
    except BlockingIOError:
        return
    else:
        with conn:
            print(f"[POSE-SERVER] connection from {addr}, sending pose #{idx}")
            try:
                conn.sendall(line.encode())
            except (ConnectionResetError, BrokenPipeError) as e:
                print(f"[POSE-SERVER] warning: клиент {addr} закрыл соединение: {e}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # SBG
    net = GraspNet_MSCQ(
        input_feature_dim=0, num_view=args.num_view,
        num_angle=12, num_depth=4, cylinder_radius=0.08,
        hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04],
        is_training=False, obs=False).to(device)
    state = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(state['model_state_dict']);  net.eval()

    # YOLO Seg
    seg_model = YOLOv8Seg(
        onnx_model='/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/best.onnx',
        yaml_path ='/home/nikita/diplom/Scale-Balanced-Grasp/yolo_module/merged_yolo_dataset2.yaml')

    # RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16,30)
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8,30)
    profile  = pipeline.start(cfg)
    align    = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color)\
                  .as_video_stream_profile().get_intrinsics()

    vis = create_open3d_visualizer('SBG viewer', 1280, 720)
    
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            dfrm, cfrm = frames.get_depth_frame(), frames.get_color_frame()
            if not dfrm or not cfrm:
                continue

            depth_img = np.asanyarray(dfrm.get_data())
            color_img = np.asanyarray(cfrm.get_data())

            # ── Segmentation ─────────────────────────────────────────
            boxes, segments, _ = seg_model(color_img, 0.6, 0.5)
            seg_vis = seg_model.draw_and_visualize(
                color_img.copy(), boxes, segments, vis=False, save=False)

            masks = []
            for poly in segments:
                m = np.zeros(depth_img.shape, np.uint8)
                cv2.fillPoly(m, [poly.astype(np.int32)], 1)
                masks.append(m.astype(bool))
            
            cv2.imshow('Segmentation', seg_vis)
            
            # ── Point cloud ──────────────────────────────────────────
            z = depth_img.astype(np.float32) * depth_scale
            h, w = depth_img.shape
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            X = (xs - intr.ppx) / intr.fx * z
            Y = (ys - intr.ppy) / intr.fy * z
            valid = z > 0
            xyz_full  = np.stack([X[valid], Y[valid], z[valid]], -1)
            color_full = color_img[valid]

            rng = (xyz_full[:,2] > 0.2) & (xyz_full[:,2] < 1.2)
            xyz_full, color_full = xyz_full[rng], color_full[rng]

            xyz_s, _ = sample_point_cloud(
                xyz_full, color_full.astype(np.float32)/255, 20000)
            if xyz_s is None:
                continue

            # ── SBG inference ───────────────────────────────────────
            pc_t = torch.from_numpy(xyz_s).float().unsqueeze(0).to(device)
            with torch.no_grad():
                end_points  = net({'point_clouds': pc_t})
                gg_offset   = pred_decode(end_points)[0].cpu().numpy()   # (N_grasp, 17)

            # ── извлечение центров грипера из gg_offset и проекция 2D ──
            centers3d = gg_offset[:, 13:16]                  # (N_grasp, 3)
            h_img, w_img = color_img.shape[:2]
            u = (centers3d[:,0] * fx / centers3d[:,2] + ppx).astype(int)
            v = (centers3d[:,1] * fy / centers3d[:,2] + ppy).astype(int)

            for ui, vi in zip(u, v):
                if 0 <= ui < w_img and 0 <= vi < h_img:
                    cv2.circle(seg_vis, (ui, vi), 5, (0, 255, 0), -1)

            #cv2.imshow('Segmentation', seg_vis)

            # ── Collision check (optional) ──────────────────────────
            if args.collision_thresh>0 and gg_offset.shape[0]:
                det  = ModelFreeCollisionDetector(
                           xyz_full, voxel_size=args.voxel_size)
                mask = det.detect(GraspGroup(gg_offset), 0.05, args.collision_thresh)
                #gg_offset   = gg_offset[~mask]

            # ── bbox-фильтрация ────────────────────────────────────
            bbox_list = []
            inside = np.zeros(len(centers3d), dtype=bool)
            for m in masks:
                bb = mask_to_bbox3d(
                    m, depth_img, intr, depth_scale,
                    depth_pad=args.bbox_depth_pad,
                    xy_pad=args.bbox_xy_pad)
                if bb is None:
                    continue
                bbox_list.append(bb)
                inside |= filter_grasps_in_bbox(centers3d, bb)
            
            # добавляем фильтр по ширине: не больше 60 мм
            widths = gg_offset[:, 1]  # столбец ширины в метрах
            #print(widths)
            inside &= (widths <= 0.06)

            # теперь отфильтруем массивы
            #gg_offset = gg_offset[inside]
            centers3d = centers3d[inside]

            # пересчитаем 2D-проекцию уже отфильтрованных центров
            u = (centers3d[:,0] * fx / centers3d[:,2] + ppx).astype(int)
            v = (centers3d[:,1] * fy / centers3d[:,2] + ppy).astype(int)
            for ui, vi in zip(u, v):
                if 0 <= ui < w_img and 0 <= vi < h_img:
                    cv2.circle(seg_vis, (ui, vi), 5, (0, 255, 0), -1)
            
            xyz_vis, col_vis = xyz_full, color_full/255.
            
            # ── отправляем захваты на сервер ─────────────────────────
            #if gg_offset.shape[0] > 0:
            send_poses(gg_offset)

            # ── визуализация ─────────────────────────────────────────
            visualize_frame_and_grasps(
                vis, xyz_vis, col_vis, gg_offset,
                bbox_list=bbox_list,
                show_axes=args.show_axes,
                show_bbox=args.show_bbox,
                max_grasp_num=args.max_grasp_num)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop();  vis.destroy_window();  cv2.destroyAllWindows()
        # закрываем сервер
        if hasattr(send_poses, "server"):
            send_poses.server.close()

if __name__ == '__main__':
    main()
