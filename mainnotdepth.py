import argparse
import sys
import time
import cv2
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO

ESC_KEY = 27


def load_rknn_model():
    return YOLO("weights/bestn_rknn_model/bestn_rknn_model")


def run_inference(rknn, image):
    input_data = cv2.resize(image, (640, 640))  # Размер входа для модели
    detections = rknn(input_data)
    return detections  # Вернём сырые выходные данные


def draw_detections(image, detections):
    if not detections:
        return image

    for det in detections[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Координаты бокса
        conf = float(det.conf[0])  # Доверие
        cls = int(det.cls[0])  # Класс
        label = f"{detections[0].names[cls]}: {conf:.2f}"

        # Рисуем бокс
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                detections = run_inference(rknn, color_image)
                last_infer_time = time.time()
                print("Detections:", detections)

            # Отрисовка детекций
            color_image = draw_detections(color_image, detections)

            cv2.imshow("YOLO Output", color_image)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
