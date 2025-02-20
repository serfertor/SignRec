from ultralytics import YOLO

# Load the exported RKNN model
rknn_model = YOLO("weights/best-rk3588.rknn")

# Run inference
results = rknn_model("Signs-2/test/images/IMG20240303224805_BURST000_COVER_jpg.rf.539093a4695890fc942797dc3b87dd70.jpg")

print(results[0])