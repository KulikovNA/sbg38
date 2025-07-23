# tcp_utils.py

import socket
import queue
import time

from save_utils import save_kept_grasp

"""
Суть работы скрипта:
- Функция tcp_sender устанавливает TCP-соединение с сервером по заданным IP и порту.
- После успешного подключения функция непрерывно извлекает данные из очереди (grasp_queue). Каждый элемент очереди – это словарь с данными, включающий:
    • "json_data": JSON-строка с информацией о захвате, отправляемая на сервер.
    • "image": цветное изображение кадра.
    • "grasp_info": словарь с информацией о захвате (например, оценка score).
    • "frame_index": индекс текущего кадра.
    • "intrinsics": параметры камеры.
    • Опционально "pointcloud": облако точек.
- Полученные данные отправляются по TCP-соединению, после чего вызывается функция save_kept_grasp для сохранения данных на диск.
- При возникновении ошибки подключения или потери связи функция пытается переподключиться каждые 5 секунд.
"""

def tcp_sender(server_ip, server_port, grasp_queue):
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((server_ip, server_port))
            print(f"TCP-соединение установлено с {server_ip}:{server_port}")
        except Exception as e:
            print("Ошибка подключения TCP:", e)
            sock.close()
            time.sleep(5)
            continue

        while True:
            try:
                data_dict = grasp_queue.get(timeout=1)
                json_data = data_dict["json_data"]
                sock.sendall((json_data + "\n").encode('utf-8'))

                color_img = data_dict["image"]
                grasp_info = data_dict["grasp_info"]
                save_index = data_dict["frame_index"]
                intrinsics = data_dict["intrinsics"]

                # Получаем облако точек, если есть
                pointcloud = data_dict.get("pointcloud", None)

                # Сохраняем всё вместе
                save_kept_grasp(
                    color_img,
                    grasp_info,
                    save_index,
                    intrinsics,
                    pointcloud=pointcloud,   # <-- передаём облако
                    output_dir="saved_grasps"
                )

                print(f"[OK] Отправлен кадр #{save_index} c score={grasp_info['score']:.2f} и сохранён.")
            
            except queue.Empty:
                # Если очередь пуста целую 1 секунду — просто повторяем
                continue
            except Exception as e:
                print("Ошибка при отправке по TCP:", e)
                break

        sock.close()
        print("TCP-соединение разорвано. Попытка переподключения...")
        time.sleep(5)

