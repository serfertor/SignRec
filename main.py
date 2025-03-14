import argparse
import sys
import time

import cv2
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO

ESC_KEY = 27

# Список классов жестов
GESTURE_CLASSES = [
    "bad", "down", "goat", "good", "heart", "jumbo",
    "ok", "paper", "rock", "scissors", "up"
]

# ID жестов
TRIGGER_GESTURE = 5  # "jumbo" – фиксируем руку
UNLOCK_GESTURE = 4  # "heart" – снимаем лок

# Данные отслеживаемой руки
tracked_hand = None  # (x1, y1, x2, y2, gesture_id)
DEPTH_THRESHOLD = 200  # Порог допустимой разницы глубины в мм


def load_rknn_model():
    """ Загружает RKNN-модель YOLO. """
    rknn = YOLO("weights/bestn_rknn_model/bestn_rknn_model")
    return rknn


def get_average_depth(depth_data, x1, y1, x2, y2):
    """ Возвращает среднюю глубину внутри указанного бокса. """
    region = depth_data[y1:y2, x1:x2]
    nonzero_values = region[region > 0]  # Игнорируем нулевые значения
    if len(nonzero_values) == 0:
        return None
    return int(np.mean(nonzero_values))  # Усредняем глубину


def process_detections(detections, image, depth_data):
    """ Обрабатывает выходные данные YOLO с учётом глубины. """
    global tracked_hand

    if not detections:
        return image

    new_tracked_hand = None

    for det in detections[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Координаты бокса
        conf = det.conf[0].item()  # Вероятность детекции
        cls = int(det.cls[0].item())  # ID класса

        label = f"{GESTURE_CLASSES[cls]}: {conf:.2f}"
        print("Detect ", label)
        avg_depth = get_average_depth(depth_data, x1, y1, x2, y2)
        if avg_depth is None:
            continue

        if tracked_hand:
            tx1, ty1, tx2, ty2, t_cls = tracked_hand
            tracked_depth = get_average_depth(depth_data, tx1, ty1, tx2, ty2)

            if tracked_depth is None or abs(avg_depth - tracked_depth) > DEPTH_THRESHOLD:
                continue  # Игнорируем, если жест далеко по глубине

            if cls == UNLOCK_GESTURE:
                print("Unlocking hand tracking...")
                tracked_hand = None
                continue

            if abs(x1 - tx1) < 50 and abs(y1 - ty1) < 50:
                new_tracked_hand = (x1, y1, x2, y2, cls)
                color = (0, 0, 255)

        elif cls == TRIGGER_GESTURE:
            print("Hand locked for tracking!")
            tracked_hand = (x1, y1, x2, y2, cls)

    if new_tracked_hand:
        tracked_hand = new_tracked_hand

    return image


def process_depth_frame(depth_frame):
    """ Обрабатывает кадр глубины и создаёт визуализацию. """
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    # Получаем данные глубины в виде массива
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))
    depth_data = depth_data.astype(np.uint16)

    # Применяем нормализацию для визуализации
    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    return depth_image


def main(argv):
    pipeline = Pipeline()
    device = pipeline.get_device()
    device_info = device.get_device_info()
    device_pid = device_info.get_pid()
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="align mode, HW=hardware mode,SW=software mode,NONE=disable align",
                        type=str, default='HW')
    parser.add_argument("-s", "--enable_sync", help="enable sync", type=bool, default=True)
    args = parser.parse_args()

    align_mode = args.mode
    enable_sync = args.enable_sync

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    except Exception as e:
        print(e)
        return

    if align_mode == 'HW':
        config.set_align_mode(OBAlignMode.HW_MODE)
    elif align_mode == 'SW':
        config.set_align_mode(OBAlignMode.SW_MODE)
    else:
        config.set_align_mode(OBAlignMode.DISABLE)

    if enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(e)

    try:
        pipeline.start(config)
    except Exception as e:
        print(e)
        return

    rknn = load_rknn_model()
    last_infer_time = time.time()
    detections = []

    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            # Обрабатываем карту глубины
            depth_image = process_depth_frame(depth_frame)

            # Получаем сырые данные глубины
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

            # Инференс раз в секунду
            if time.time() - last_infer_time > 1:
                detections = rknn(color_image)
                last_infer_time = time.time()

            color_image = process_detections(detections, color_image, depth_data)

        except KeyboardInterrupt:
            break

    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
