import os
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train4/weights/best.pt')

image_folder='D:/杂七杂八/学校的文件/毕业设计/数据集/split_CTIR_Dataset/test/images/'

# 检测图像
results = model(image_folder)
save_dir='D:/杂七杂八/学校的文件/毕业设计/数据集/split_CTIR_Dataset/test/images/testresults/'
os.makedirs(save_dir, exist_ok=True)
# 保存结果
results = model(image_folder, save=True, save_dir=save_dir)

# 解析结果
# for result in results:
#     boxes = result.boxes
#     classes = result.boxes.cls
#     confidences = result.boxes.conf
#
#     for box, cls, conf in zip(boxes, classes, confidences):
#         print(f"Class: {cls}, Confidence: {conf}, Box: {box}")