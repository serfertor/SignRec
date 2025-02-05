import cv2

from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile
from utils import frame_to_bgr_image
from ultralytics import YOLO
import supervision as sv

ESC_KEY = 27


def main():
    model = YOLO("best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            result = model(color_image, conf=0.6)[0]
            detections = sv.Detections.from_ultralytics(result)
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections['class_name'], detections.confidence)
            ]

            frame = label_annotator.annotate(
                scene=color_image,
                detections=detections,
                labels=labels
            )
            cv2.imshow("Color Viewer", frame)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    pipeline.stop()


'''            for data in detections.boxes.data.tolist():
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)'''

if __name__ == "__main__":
    main()
