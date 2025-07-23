# save_utils.py
import os
import json
import cv2
import open3d as o3d
import numpy as np

"""
Суть работы скрипта:
- Предоставляет утилиты для сохранения результатов захвата (grasp) и облаков точек.
- Функция save_kept_grasp:
    • Сохраняет цветное изображение кадра в формате PNG.
    • Сохраняет информацию о захвате (grasp_info) и параметры камеры (intrinsics) в JSON.
    • (Опционально) Сохраняет облако точек в формате PCD, если оно передано.
    • Формирует 17-элементный массив с данными захвата (score, заполнители, матрица вращения, вектор переноса, obj_id) и сохраняет его в формате NPY.
- Функция save_pointcloud:
    • Сохраняет облако точек (и опционально цвета) в формате PCD.
"""

def save_kept_grasp(
    color_img,
    grasp_info,
    frame_index,
    intrinsics,
    pointcloud=None,
    output_dir="saved_grasps"
):
    """
    Сохраняем файлы для кадра:
      - Картинку (frame_XXXXXX.png)
      - JSON (grasp_info_XXXXXX.json)
      - (опционально) облако (cloud_XXXXXX.pcd)
      - Дополнительно: 17-элементный массив (grasp_XXXXXX.npy), 
        чтобы потом можно было загрузить через GraspGroup (при желании).

    Параметры:
      color_img   : np.array (H,W,3) BGR
      grasp_info  : dict с ключами 'score', 'rotation'(9), 'translation'(3), ...
      frame_index : int
      intrinsics  : dict (если нужно, для JSON)
      pointcloud  : (N,3) или (N,6) (XYZ или XYZRGB), если не None, сохраним .pcd
      output_dir  : папка для сохранения
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1) Сохраняем цветную картинку
    img_path = os.path.join(output_dir, f"frame_{frame_index:06d}.png")
    cv2.imwrite(img_path, color_img)

    # 2) Сохраняем JSON (как раньше)
    json_dict = {
        "frame_index": frame_index,
        "grasp_info": grasp_info, 
        "intrinsics": intrinsics
    }
    json_path = os.path.join(output_dir, f"grasp_info_{frame_index:06d}.json")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)

    # 3) Сохраняем облако (опционально)
    if pointcloud is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])

        if pointcloud.shape[1] >= 6:
            colors = pointcloud[:, 3:6].astype(float32)
            if colors.max() > 1.1:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_path = os.path.join(output_dir, f"cloud_{frame_index:06d}.pcd")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"[SAVE] Сохранено облако: {pcd_path}")

    # 4) Дополнительно: делаем 17-элементный массив (GraspNet-формат)
    #    Пример: [score, width, height, depth, rotation(9), translation(3), obj_id(1)]
    #    Если у вас нет width/height/depth/obj_id в grasp_info — ставим заглушки (0).
    #    rotation — это список из 9 float,
    #    translation — список из 3 float.

    # Извлекаем поля из grasp_info (score, rotation, translation)
    score = float(grasp_info.get("score", 0.0))
    rot_list = grasp_info.get("rotation", [0]*9)      # 9 значений
    trans_list = grasp_info.get("translation", [0]*3) # 3 значения

    # Допустим, нет width/height/depth/object_id => ставим 0
    width  = 0.0
    height = 0.0
    depth  = 0.0
    obj_id = 0.0

    # Склеиваем в numpy
    grasp_17 = np.zeros((17,), dtype=float32)
    grasp_17[0] = score
    grasp_17[1] = width
    grasp_17[2] = height
    grasp_17[3] = depth
    grasp_17[4:13]  = rot_list  # 9 чисел
    grasp_17[13:16] = trans_list
    grasp_17[16]    = obj_id

    # Сохраним как grasp_XXXXXX.npy
    grasp_npy_path = os.path.join(output_dir, f"grasp_{frame_index:06d}.npy")
    np.save(grasp_npy_path, grasp_17)

    print(f"[save_kept_grasp] Кадр #{frame_index}:")
    print(f"   PNG  => {img_path}")
    print(f"   JSON => {json_path}")
    print(f"   NPY  => {grasp_npy_path}")


def save_pointcloud(pointcloud, colors, frame_index, output_dir="saved_pointclouds"):
    """
    Сохраняет облако точек в формате .pcd с заданным индексом кадра.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    # Если переданы цвета, добавляем их (нормализуем в диапазон [0,1], если нужно)
    if colors is not None:
        colors_arr = colors.astype(float32)
        if colors_arr.max() > 1.1:  
            colors_arr /= 255.0      # приводим к 0-1, если это 0-255
        pcd.colors = o3d.utility.Vector3dVector(colors_arr)
    file_path = os.path.join(output_dir, f"cloud_{frame_index}.pcd")
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"[SAVE] Облако точек сохранено: {file_path}")