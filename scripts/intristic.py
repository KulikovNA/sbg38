import pyrealsense2 as rs

# Создаем объект конвейера
pipeline = rs.pipeline()

# Настраиваем конфигурацию
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Запускаем поток
pipeline.start(config)

try:
    # Получаем кадр
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        raise ValueError("Не удалось получить depth frame")

    # Получаем профиль и интринсики
    profile = depth_frame.profile
    intrinsics = profile.as_video_stream_profile().intrinsics

    print("Интринсики камеры:")
    print(f"Фокусное расстояние по X (fx): {intrinsics.fx}")
    print(f"Фокусное расстояние по Y (fy): {intrinsics.fy}")
    print(f"Оптический центр по X (cx): {intrinsics.ppx}")
    print(f"Оптический центр по Y (cy): {intrinsics.ppy}")
    print(f"Коэффициенты искажения: {intrinsics.coeffs}")
    print(f"Разрешение: {intrinsics.width}x{intrinsics.height}")

finally:
    # Останавливаем поток
    pipeline.stop()
