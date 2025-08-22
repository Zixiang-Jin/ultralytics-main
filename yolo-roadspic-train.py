from ultralytics import YOLO
model=YOLO('yolov8n.pt')
model.train(data='yolo-roadspic-2.yaml', epochs=50, batch=16, workers=0)