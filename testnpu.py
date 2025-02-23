from ultralytics import YOLO

# Load the exported RKNN model
rknn_model = YOLO("weights/bestn_rknn_model/bestn_rknn_model")
model = YOLO("weights/best (1).pt")

# Run inference
results = rknn_model("test.jpg")

results1 = model("test.jpg")

print(results[0])

print(results1[0])