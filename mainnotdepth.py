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

# Параметры отслеживания
TRIGGER_GESTURE = "ok"  # Жест для фиксации руки
UNLOCK_GESTURE = "down"  # Жест для сброса отслеживания
TRACKING_RADIUS = 100  # Радиус области отслеживания


def load_rknn_model():
    """ Загружает RKNN-модель YOLO. """
    rknn = YOLO("weights/bestn_rknn_model/bestn_rknn_model")
    return rknn


def process_detections(detections, image, tracking_point):
    """ Обрабатывает детекции: рисует боксы и проверяет попадание в область отслеживания. """
    if not detections:
        return image, None, None

    new_tracking_point = None
    new_tracking_class = None

    for det in detections[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Координаты бокса
        conf = det.conf[0].item()  # Вероятность детекции
        cls = int(det.cls[0].item())  # ID класса
        label = f"{GESTURE_CLASSES[cls]}: {conf:.2f}"
        color = (0, 255, 0)  # Зеленый цвет для боксов

        # Вычисляем центр бокса
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Проверяем, попадает ли бокс в область отслеживания
        if tracking_point:
            dist = np.sqrt((center_x - tracking_point[0]) ** 2 + (center_y - tracking_point[1]) ** 2)
            if dist <= TRACKING_RADIUS:
                new_tracking_point = (center_x, center_y)
                new_tracking_class = GESTURE_CLASSES[cls]
                color = (255, 0, 0)  # Отмечаем отслеживаемую руку синим

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Если триггер-жест, фиксируем руку
        if GESTURE_CLASSES[cls] == TRIGGER_GESTURE:
            new_tracking_point = (center_x, center_y)
            new_tracking_class = GESTURE_CLASSES[cls]

    return image, new_tracking_point, new_tracking_class


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
    tracking_point = None  # Точка центра отслеживаемой руки
    tracking_class = None  # Последний распознанный жест отслеживаемой руки

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

                # Обработка детекций и обновление точки отслеживания
                color_image, new_tracking_point, new_tracking_class = process_detections(detections, color_image,
                                                                                         tracking_point)

                # Если получен триггер-жест, фиксируем руку
                if new_tracking_class == TRIGGER_GESTURE:
                    tracking_point = new_tracking_point
                    print("Триггер-жест зафиксирован, отслеживаем руку")

                # Если жест сброса — убираем отслеживание
                if new_tracking_class == UNLOCK_GESTURE:
                    tracking_point = None
                    print("Отслеживание руки сброшено")

            cv2.imshow("YOLO Tracking", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
