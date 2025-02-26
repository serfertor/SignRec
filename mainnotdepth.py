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
tracked_point = None  # (x, y) – центр ладони
tracking_frames = 0  # Счётчик кадров для плавного исчезновения точки
TRACKING_RADIUS = 100  # Радиус отслеживания


def load_rknn_model():
    """ Загружает RKNN-модель YOLO. """
    rknn = YOLO("weights/bestn_rknn_model/bestn_rknn_model")
    return rknn


def process_detections(detections, image):
    """ Обрабатывает выходные данные YOLO и рисует боксы. """
    global tracked_point, tracking_frames

    if not detections:
        return image

    new_tracked_point = None

    for det in detections[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Координаты бокса
        conf = det.conf[0].item()  # Вероятность детекции
        cls = int(det.cls[0].item())  # ID класса

        label = f"{GESTURE_CLASSES[cls]}: {conf:.2f}"
        color = (0, 255, 0)  # Зеленый цвет для боксов

        # Вычисляем центр бокса (ладони)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Если уже отслеживаем руку, проверяем, что детекция рядом
        if tracked_point:
            px, py = tracked_point

            # Если найден жест снятия блока (heart) — сбрасываем отслеживание
            if cls == UNLOCK_GESTURE:
                print("Unlocking hand tracking...")
                tracked_point = None
                tracking_frames = 0
                continue

            # Если рука в радиусе отслеживания — обновляем точку
            dist = np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2)
            if dist <= TRACKING_RADIUS:
                new_tracked_point = (center_x, center_y)
                color = (0, 0, 255)  # Красный цвет для отслеживаемой руки

        # Если триггер-жест (jumbo) и нет отслеживаемой руки — фиксируем
        elif cls == TRIGGER_GESTURE:
            print("Hand locked for tracking!")
            tracked_point = (center_x, center_y)
            tracking_frames = 10  # Устанавливаем таймер на 5 кадров
            color = (0, 0, 255)  # Красный цвет для отслеживаемой руки

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Обновляем точку отслеживания
    if new_tracked_point:
        tracked_point = new_tracked_point
        tracking_frames = 5  # Обновляем таймер
    elif tracked_point:
        tracking_frames -= 1
        if tracking_frames <= 0:
            tracked_point = None  # Если 5 кадров нет руки, сбрасываем отслеживание

    return image


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

            # Инференс раз в секунду
            if time.time() - last_infer_time > 1:
                detections = rknn(color_image)
                last_infer_time = time.time()

            color_image = process_detections(detections, color_image)

            cv2.imshow("YOLO Output", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
