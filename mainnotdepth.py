import cv2
import numpy as np
import pyorbbecsdk
import time
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

frame_count = 0
start_time = time.time()

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

    # Подсчёт FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        print(f"FPS: {frame_count / elapsed_time:.2f}")
        frame_count = 0
        start_time = time.time()

    # Отображение кадра
    cv2.imshow("Color Frame", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Остановка пайплайна и закрытие окна
pipeline.stop()
cv2.destroyAllWindows()
