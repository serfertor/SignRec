from ultralytics import YOLO

# Load the exported RKNN model
rknn_model = YOLO("weights/bestn_rknn_model/bestn_rknn_model")

# Run inference
results = rknn_model("test.jpg")

print(results[0])
