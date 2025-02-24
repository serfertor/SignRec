import cv2
import numpy as np
import time
import pyorbbecsdk
from pyorbbecsdk import Pipeline, Config, OBSensorType
from utils import frame_to_bgr_image

# Создаём объект пайплайна
pipeline = Pipeline()

# Получаем устройство и его информацию
device = pipeline.get_device()
device_info = device.get_device_info()
print(f"Подключено устройство: {device_info.get_name()}")

# Конфигурируем поток цветного изображения
config = Config()
color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
config.enable_stream(color_profile)

# Запускаем поток
pipeline.start(config)

prev_time = time.time()
frame_count = 0
fps = 0

while True:
    frames = pipeline.wait_for_frames(100)  # Ожидание кадров (таймаут 100 мс)
    if frames is None:
        print("Ошибка: Не получены кадры")
        continue

    color_frame = frames.get_color_frame()
    if color_frame is None:
        print("Ошибка: Нет цветного кадра")
        continue

    # Конвертация кадра в OpenCV формат (BGR)
    color_image = frame_to_bgr_image(color_frame)

    # Вычисление FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        prev_time = current_time

    # Отображение FPS
    cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Рисуем прямоугольник в центре кадра
    height, width, _ = color_image.shape
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4
    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Отображение кадра
    cv2.imshow("Color Frame", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Остановка пайплайна и закрытие окна
pipeline.stop()
cv2.destroyAllWindows()
