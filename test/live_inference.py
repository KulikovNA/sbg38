# live_inference.py

import os
import time
import numpy as np
import queue
import threading
import open3d as o3d

import pyrealsense2 as rs

from config import cfgs
from realsense_utils import get_realsense_pointcloud, preprocess_pointcloud
from inference_utils import init_model, run_inference, filter_collisions
from visualization_utils import draw_grasps, custom_color_map
from tcp_utils import tcp_sender

from graspnetAPI.grasp import GraspGroup
from collision_detector import ModelFreeCollisionDetector

def is_valid_grasp(grasp):
    """
    Проверка, подходит ли захват под ваши критерии.
    grasp: numpy.array формы (19,) (или сколько у вас полей).
      - grasp[0] = score
      - grasp[4:13] = rotation (3x3, 9 чисел)
      - grasp[13:16] = translation (x, y, z)
    Возвращаем True/False.
    """
    translation = grasp[13:16]
    x, y, z = translation
    dist = np.linalg.norm(translation)
    # Пример фильтра: x,y по -0.3..0.3, z по 0.2..1.0, distance <= 1.2
    if -0.15 < x < 0.15 and -0.3 < y < 0.3 and 0.2 < z < 0.55 and dist < 1.2:
        print(f"Grasp at x={x:.2f}, y={y:.2f}, z={z:.2f}, dist={dist:.2f} => VALID")
        return True
    return False

def live_inference():
    """
    1) Один раз получаем облако с RealSense.
    2) Запускаем инференс, берём все захваты.
    3) Применяем фильтрацию (is_valid_grasp).
    4) Из отфильтрованных выбираем лучший (по score).
    5) Отправляем лучший по TCP (и сохраняем).
    6) Отображаем результат в окне O3D, ждём, пока пользователь не закроет окно.
    7) Завершаем скрипт.
    """

    # --- Инициализация модели
    net = init_model()

    # --- Настройка RealSense
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    # -- Пропускаем 50 кадров --
    for i in range(50):
        pipeline.wait_for_frames()
    print("[INFO] Пропустили 50 кадров, начинаем реальную обработку.")
    
    # --- Очередь и поток TCP
    grasp_queue = queue.Queue()
    server_ip = "127.0.0.1"
    server_port = 12345
    sender_thread = threading.Thread(
        target=tcp_sender,
        args=(server_ip, server_port, grasp_queue),
        daemon=True
    )
    sender_thread.start()

    # --- Коллиз-детектор
    collision_detector = [None]

    # --- Визуализатор
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='GraspNet Single Inference', width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(axes)

    num_grasps_to_display = 30
    gripper_geometries = []
    far_away = np.array([[1000, 1000, 1000]] * 4)
    for _ in range(num_grasps_to_display):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(far_away)
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0,1],[1,2],[2,3],[3,0]]))
        line_set.colors = o3d.utility.Vector3dVector(
            [custom_color_map(0.0) for _ in range(4)]
        )
        gripper_geometries.append(line_set)
        vis.add_geometry(line_set)

    vis.poll_events()
    vis.update_renderer()

    # Параметры камеры (если нужно)
    camera_params_file = "camera_params.json"
    ctr = vis.get_view_control()
    if os.path.exists(camera_params_file):
        try:
            param = o3d.io.read_pinhole_camera_parameters(camera_params_file)
            ctr.convert_from_pinhole_camera_parameters(param)
            print(f"Параметры камеры загружены из {camera_params_file}")
        except Exception as e:
            print(f"Не удалось загрузить параметры камеры: {e}")
    else:
        ctr.set_front([1, -1, 1])
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_zoom(0.8)

    def save_camera_callback(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(camera_params_file, param)
        print(f"Параметры камеры сохранены в {camera_params_file}")
        return False

    vis.register_key_callback(ord('S'), save_camera_callback)

    try:
        # 1) Попытки взять облако
        max_tries = 10
        pointcloud_raw, colors_raw, extra_data = None, None, None

        for i in range(max_tries):
            vtx, cols, edata = get_realsense_pointcloud(pipeline, pc, align)
            if vtx is not None and len(vtx) > 0:
                pointcloud_raw, colors_raw, extra_data = vtx, cols, edata
                break
            print(f"[WARN] Облако точек пустое, повтор {i+1}/{max_tries}...")
            time.sleep(0.2)

        if pointcloud_raw is None or len(pointcloud_raw) == 0:
            print("[ERROR] Не удалось получить облако точек. Выходим.")
            return

        (intr, color_img) = extra_data

        # 2) Предобработка
        pointcloud_processed, colors_processed = preprocess_pointcloud(
            pointcloud_raw, colors_raw, num_points=cfgs.num_point
        )
        if pointcloud_processed is None:
            print("[ERROR] Недостаточно точек после предобработки. Выходим.")
            return

        pcd.points = o3d.utility.Vector3dVector(pointcloud_processed)
        pcd.colors = o3d.utility.Vector3dVector(colors_processed)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # 3) Инференс
        gg = run_inference(net, pointcloud_processed)

        # 4) Фильтр коллизий
        gg = filter_collisions(gg, pointcloud_processed, collision_detector)

        if len(gg) == 0:
            print("[INFO] Нет подходящих захватов (после коллизий). Выходим.")
            return

        # 5) Сортируем ВСЕ (не только top_k)
        gg.sort_by_score()

        # 5а) Делаем дополнительную фильтрацию
        print(f"Всего захватов после коллизий: {len(gg)}")
        all_grasps = gg.grasp_group_array  # shape (N, 19)
        filtered = [g for g in all_grasps if is_valid_grasp(g)]
        print(f"Прошло фильтр: {len(filtered)}")
        if len(filtered) == 0:
            print("[INFO] Нет захватов, прошедших фильтр (координаты/дистанция). Выходим.")
            return

        # Сортируем отфильтрованные по score (ещё раз, на всякий случай)
        filtered = np.array(filtered)
        scores = filtered[:, 0]
        sort_idx = np.argsort(scores)[::-1]
        filtered = filtered[sort_idx]

        best_grasp_array = filtered[0]
        best_score = float(best_grasp_array[0])
        best_rotation = best_grasp_array[4:13].tolist()
        best_translation = best_grasp_array[13:16].tolist()

        # 5б) Для визуализации top_gg возьмём первые num_grasps_to_display из filtered
        top_count = min(num_grasps_to_display, len(filtered))
        top_filtered = filtered[:top_count]
        # пересоздадим новый GraspGroup, чтобы draw_grasps работал как обычно
        top_gg = GraspGroup(np.array(top_filtered))

        # Рисуем top_gg
        draw_grasps(top_gg, gripper_geometries, vis)
        vis.poll_events()
        vis.update_renderer()

        # 6) Формируем JSON (только лучший)
        import json
        best_grasp_list = best_grasp_array.tolist()
        json_str = json.dumps(best_grasp_list)

        data_dict = {
            "json_data": json_str,
            "image": color_img,
            "frame_index": 0,
            "intrinsics": {
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy
            },
            "grasp_info": {
                "score": best_score,
                "rotation": best_rotation,
                "translation": best_translation
            },
            "pointcloud": pointcloud_processed
        }

        # 7) Отправляем
        grasp_queue.put(data_dict)
        print(f"[SINGLE] Отправлен лучший захват после фильтра (score={best_score:.2f}).")

        # Ждём, пока пользователь закроет окно
        print("[INFO] Окно Open3D показывает отфильтрованные схваты. Закройте окно для завершения.")
        while vis.poll_events():
            vis.update_renderer()
            time.sleep(0.05)

    finally:
        pipeline.stop()
        vis.destroy_window()
        print("[INFO] Завершение скрипта.")
