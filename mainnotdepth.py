import cv2
import numpy as np
import pyorbbecsdk
from pyorbbecsdk import Pipeline, Config, OBSensorType
from utils import frame_to_bgr_image
import time

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

while True:
    frames = pipeline.wait_for_frames(300)  # Ожидание кадров (таймаут 100 мс)
    if frames is None:
        print("Ошибка: Не получены кадры")
        continue

    color_frame = frames.get_color_frame()
    if color_frame is None:
        print("Ошибка: Нет цветного кадра")
        continue

    # Конвертация кадра в OpenCV формат (BGR)
    color_image = frame_to_bgr_image(color_frame)

    # Отображение кадра
    cv2.imshow("Color Frame", color_image)

    # Явно освобождаем переменные с кадрами
    del color_frame
    del frames

    # Добавляем небольшую задержку для снижения нагрузки
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Остановка пайплайна и закрытие окна
pipeline.stop()
cv2.destroyAllWindows()
