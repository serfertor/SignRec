from ultralytics import YOLO

model = YOLO('C:/Users/Sherri/PycharmProjects/SignRec/weights/best (1).pt')
path = model.export(format='rknn')
