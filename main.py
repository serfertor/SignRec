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

depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
config.enable_stream(depth_profile)

print(f"Color profile: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()} FPS")
print(f"Depth profile: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()} FPS")

config.set_align_mode(OBAlignMode.HW_MODE)
pipeline.enable_frame_sync()
pipeline.start(config)

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


def detect_and_track(color_frame, depth_frame):
    global tracked_hand, tracked_trajectory, last_inference_time, missed_frames, last_tracked_bbox, last_tracked_trajectory

    current_time = time.time()
    if current_time - last_inference_time >= 1:  # Инференс раз в секунду
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
                    depth_value = depth_frame[y1:y2, x1:x2].mean()
                    tracked_hand = {"bbox": hand["bbox"], "depth": depth_value}
                    tracked_trajectory = []
                    missed_frames = 0
                    last_tracked_bbox = hand["bbox"]  # Кэшируем бокс
                    last_tracked_trajectory = []  # Очищаем траекторию
                    print(
                        f"✅ Захвачена рука с жестом {TRIGGER_GESTURE} - Координаты: {x1, y1, x2, y2}, Глубина: {depth_value:.2f}")
                    break
        else:
            # Поиск самой близкой руки к сохранённой глубине
            closest_hand = None
            min_depth_diff = float("inf")

            for hand in hands:
                x1, y1, x2, y2 = hand["bbox"]
                depth_value = depth_frame[y1:y2, x1:x2].mean()

                depth_diff = abs(depth_value - tracked_hand["depth"])
                if depth_diff < min_depth_diff:
                    min_depth_diff = depth_diff
                    closest_hand = hand

            if closest_hand and min_depth_diff < 400 and closest_hand["label"] != RESET_GESTURE:
                tracked_hand["bbox"] = closest_hand["bbox"]
                tracked_hand["depth"] = depth_frame[closest_hand["bbox"][1]:closest_hand["bbox"][3],
                                        closest_hand["bbox"][0]:closest_hand["bbox"][2]].mean()
                x1, y1, x2, y2 = tracked_hand["bbox"]
                tracked_trajectory.append(((x1 + x2) // 2, (y1 + y2) // 2))
                missed_frames = 0

                last_tracked_bbox = tracked_hand["bbox"]  # Кэшируем бокс
                last_tracked_trajectory = tracked_trajectory  # Кэшируем траекторию

                print(f"🔵 Отслеживаем руку - Координаты: {x1, y1, x2, y2}, Глубина: {tracked_hand['depth']:.2f}")
                '''
                здесь можно прописать кейсы использовангия жестов
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
    frames: FrameSet = pipeline.wait_for_frames(500)
    if frames is None:
        return None, None

    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None, None
    color_image = frame_to_bgr_image(color_frame)

    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None, None
    depth_data = (np.frombuffer(depth_frame.get_data().copy(order='C'), dtype=np.uint16).copy(order='C')
                  .reshape((depth_frame.get_height(), depth_frame.get_width())))

    return color_image, depth_data


def visualize_depth(depth_frame):
    depth_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colormap


while True:
    color_frame, depth_frame = get_frames()
    if color_frame is None or depth_frame is None:
        continue

    processed_frame = detect_and_track(color_frame, depth_frame)
    depth_colormap = visualize_depth(depth_frame)

    combined_view = np.hstack((processed_frame, depth_colormap))

    cv2.imshow('Sign Recognition', combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
