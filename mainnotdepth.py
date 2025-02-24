import cv2
import numpy as np
import pyorbbecsdk
import time
from pyorbbecsdk import Pipeline, Config, OBSensorType
from utils import frame_to_bgr_image
from ultralytics import YOLO

# Инициализация камеры
pipeline = Pipeline()
device = pipeline.get_device()
device_info = device.get_device_info()
print(f"Подключено устройство: {device_info.get_name()}")

# Настройка потока цветного изображения
config = Config()
color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
config.enable_stream(color_profile)

# Запуск камеры
pipeline.start(config)

# Загрузка RKNN модели YOLO
model = YOLO("weights/bestn_rknn_model/bestn_rknn_model")  # bestn (nano) или best (small)

# Таймер для инференса (1 кадр в секунду)
last_inference_time = 0

while True:
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        print("Ошибка: Не получены кадры")
        continue

    color_frame = frames.get_color_frame()
    if color_frame is None:
        print("Ошибка: Нет цветного кадра")
        continue

    # Конвертация кадра в OpenCV BGR
    color_image = frame_to_bgr_image(color_frame)

    # Инференс раз в секунду
    current_time = time.time()
    if current_time - last_inference_time >= 1:
        last_inference_time = current_time
        results = model(cv2.resize(color_image, (640, 640)))

        # Отрисовка боксов
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                conf = float(box.conf)

                # Рисуем бокс
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow("YOLO Detection", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
pipeline.stop()
cv2.destroyAllWindows()
