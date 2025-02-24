import cv2
import numpy as np
import time
import argparse
import pyorbbecsdk
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode, FrameSet
from ultralytics import YOLO
from utils import frame_to_bgr_image

pipeline = Pipeline()
device = pipeline.get_device()
device_info = device.get_device_info()
device_pid = device_info.get_pid()
config = Config()

color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
config.enable_stream(color_profile)


print(f"Color profile: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()} FPS")

config.set_align_mode(OBAlignMode.HW_MODE)
pipeline.enable_frame_sync()
pipeline.start(config)

time.sleep(2)

model = YOLO("weights/bestn_rknn_model/bestn_rknn_model")  #bestn (nano) или best (small)

TRIGGER_GESTURE = "jumbo"
RESET_GESTURE = "heart"
tracked_hand = None
tracked_trajectory = []
last_inference_time = 0  # Таймер для инференса 1 кадр в секунду
max_miss_frames = 60  # Количество кадров для сброса трека
missed_frames = 0  # Счётчик пропущенных кадров
last_tracked_bbox = None  # Последний бокс руки
last_tracked_trajectory = []  # Последняя траектория движения


def detect_and_track(color_frame):
    global tracked_hand, tracked_trajectory, last_inference_time, missed_frames, last_tracked_bbox, last_tracked_trajectory

    current_time = time.time()
    if current_time - last_inference_time >= 5:  # Инференс раз в секунду
        last_inference_time = current_time
        results = model(cv2.resize(color_frame, (640, 640)))

        hands = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                conf = float(box.conf)

                hands.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": label,
                    "confidence": conf
                })

        if tracked_hand is None:
            for hand in hands:
                if hand["label"] == TRIGGER_GESTURE:
                    x1, y1, x2, y2 = hand["bbox"]
                    tracked_hand = {"bbox": hand["bbox"]}
                    tracked_trajectory = []
                    missed_frames = 0
                    last_tracked_bbox = hand["bbox"]  # Кэшируем бокс
                    last_tracked_trajectory = []  # Очищаем траекторию
                    print(
                        f"✅ Захвачена рука с жестом {TRIGGER_GESTURE} - Координаты: {x1, y1, x2, y2}")
                    break
        else:
            # Поиск самой близкой руки к сохранённой глубине
            closest_hand = None
            for hand in hands:
                x1, y1, x2, y2 = hand["bbox"]
                closest_hand = hand

            if closest_hand["label"] != RESET_GESTURE:
                tracked_hand["bbox"] = closest_hand["bbox"]
                x1, y1, x2, y2 = tracked_hand["bbox"]
                tracked_trajectory.append(((x1 + x2) // 2, (y1 + y2) // 2))
                missed_frames = 0

                last_tracked_bbox = tracked_hand["bbox"]  # Кэшируем бокс
                last_tracked_trajectory = tracked_trajectory  # Кэшируем траекторию

                print(f"🔵 Отслеживаем руку - Координаты: {x1, y1, x2, y2}")
                '''
                здесь можно прописать кейсы использования жестов
                '''

            else:
                missed_frames += 1
                if missed_frames > max_miss_frames or closest_hand["label"] != RESET_GESTURE:
                    print("❌ Рука потеряна, сбрасываем трек")
                    tracked_hand = None
                    tracked_trajectory = []
                    last_tracked_bbox = None
                    last_tracked_trajectory = []

    if last_tracked_bbox:
        x1, y1, x2, y2 = last_tracked_bbox
        cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_frame, "Tracking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i in range(1, len(last_tracked_trajectory)):
        cv2.line(color_frame, last_tracked_trajectory[i - 1], last_tracked_trajectory[i], (0, 0, 255), 2)

    return color_frame


def get_frames():
    print("⏳ Ожидание кадров от камеры...")
    frames: FrameSet = pipeline.wait_for_frames(300)
    print("✅ Кадры получены!")
    if frames is None:
        return None, None

    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None, None
    color_image = frame_to_bgr_image(color_frame)

    return color_image


while True:
    print("⏳ Запрашиваем кадры...")
    color_frame = get_frames()
    print("✅ Кадры получены!")

    if color_frame is None:
        print("⚠️ Кадры не получены, продолжаем ожидание...")
        continue

    processed_frame = detect_and_track(color_frame)

    cv2.imshow('Sign Recognition', processed_frame)
    #cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pipeline.stop()
        cv2.destroyAllWindows()
        break

pipeline.stop()
cv2.destroyAllWindows()
