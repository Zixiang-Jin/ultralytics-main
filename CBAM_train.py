import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\anaconda project\yolov8\ultralytics-main\yolo11CBAM.yaml')
    model.train(data=r'D:\anaconda project\yolov8\ultralytics-main\yolo-roadspic-3.yaml',
                cache=False,
                imgsz=640,
                epochs=50,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='cpu',
                optimizer='SGD',
                amp=True,
                project='runs/train',
                name='exp',
                )