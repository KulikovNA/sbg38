#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт сохраняет N кадров RGB и Depth из RealSense,
а также сохраняет intrinsics в JSON и в .npz вместе с цветом и глубиной.
"""
import os, json, argparse
import numpy as np
import pyrealsense2 as rs
import cv2

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir',    required=True, help='Куда сохранить кадры')
    p.add_argument('--num_frames',    type=int, default=100, help='Сколько кадров сохранить')
    p.add_argument('--skip_frames',   type=int, default=0,   help='Пропускать первые N кадров')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # запускаем камеру
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16,30)
    cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8,30)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    # читаем intrinsics и сохраняем их сразу
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    cam_info = {
        'fx': intr.fx, 'fy': intr.fy,
        'ppx': intr.ppx, 'ppy': intr.ppy,
        'width': intr.width, 'height': intr.height,
        'depth_scale': depth_sensor.get_depth_scale()
    }
    with open(os.path.join(args.output_dir, 'camera_intrinsics.json'), 'w') as f:
        json.dump(cam_info, f, indent=2)

    saved = 0
    processed = 0

    try:
        while saved < args.num_frames:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            d = aligned.get_depth_frame()
            c = aligned.get_color_frame()

            # отладка
            print(f"[DEBUG] processed={processed}, saved={saved}, depth_frame={bool(d)}, color_frame={bool(c)}")
            processed += 1

            # пропускаем первые skip_frames
            if processed <= args.skip_frames:
                continue

            if not d or not c:
                continue

            # получаем массивы
            depth = np.asanyarray(d.get_data())
            color = np.asanyarray(c.get_data())

            # генерируем имена
            name = f"{saved:04d}"
            p_color = os.path.join(args.output_dir, f'color_{name}.png')
            p_depth = os.path.join(args.output_dir, f'depth_{name}.png')
            p_npz   = os.path.join(args.output_dir, f'frame_{name}.npz')

            # сохраняем
            cv2.imwrite(p_color, color)
            cv2.imwrite(p_depth, depth)
            np.savez_compressed(p_npz,
                                color=color,
                                depth=depth,
                                intrinsics=cam_info)

            print(f"[SAVED] #{saved} → {p_color}, {p_depth}, {p_npz}")
            saved += 1

    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()
