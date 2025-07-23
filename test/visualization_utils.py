# visualization_utils.py
import open3d as o3d
import numpy as np

def custom_color_map(score):
    """
    Пример цветовой карты в зависимости от score
    """
    if score > 0.8:
        return np.array([0.0, 1.0, 0.0])  # зелёный
    elif score > 0.5:
        return np.array([1.0, 1.0, 0.0])  # жёлтый
    else:
        return np.array([1.0, 0.0, 0.0])  # красный

def draw_grasps(gg, gripper_geometries, vis, color_map=custom_color_map):
    """
    Отрисовка захватов на основе GraspGroup (gg).
    gripper_geometries — список LineSet'ов, заранее созданных и добавленных в vis.
    """
    num_geometries = len(gripper_geometries)
    num_grasps = len(gg)

    for i in range(num_geometries):
        if i < num_grasps:
            grasp = gg[i]
            rotation_matrix = grasp.rotation_matrix
            translation = grasp.translation
            width = grasp.width
            depth = grasp.depth
            score = grasp.score

            approach = rotation_matrix[:, 2]
            binormal = rotation_matrix[:, 1]
            axis_x = np.cross(approach, binormal)
            axis_x /= np.linalg.norm(axis_x)

            half_width = width / 2
            half_depth = depth / 2
            corners = [
                translation + binormal * half_width + axis_x * half_depth,
                translation - binormal * half_width + axis_x * half_depth,
                translation - binormal * half_width - axis_x * half_depth,
                translation + binormal * half_width - axis_x * half_depth,
            ]
            grasp_color = color_map(score)

            gripper_geometries[i].points = o3d.utility.Vector3dVector(corners)
            gripper_geometries[i].colors = o3d.utility.Vector3dVector(
                [grasp_color for _ in range(len(gripper_geometries[i].lines))]
            )
            vis.update_geometry(gripper_geometries[i])
        else:
            pass
