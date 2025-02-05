from roboflow import Roboflow

rf = Roboflow(api_key="VWPhbTzpYmpZqUa9VTLw")
project = rf.workspace("ozon-hzbqr").project("signs-locqo")
version = project.version(2)
dataset = version.download("yolov11")

version.deploy("yolov11", "weights", "best.pt")
