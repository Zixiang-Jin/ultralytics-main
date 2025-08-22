import cv2
from ultralytics import YOLO
import winsound

# 加载模型
model = YOLO("runs/detect/train3/weights/best.pt")  # 替换为你的模型路径

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 报警参数
target_class = "car"
alarm_threshold = 0.7
alarm_frequency = 1000  # 蜂鸣频率（Hz）
alarm_duration = 1000   # 持续时间（毫秒）

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, stream=True, verbose=False)
    alarm_triggered = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[class_id]

            if class_name == target_class and conf >= alarm_threshold:
                print(f"警报：检测到 {target_class}！置信度：{conf:.2f}")
                alarm_triggered = True

        annotated_frame = result.plot()
        cv2.imshow("Detection", annotated_frame)

    if alarm_triggered:
        winsound.Beep(alarm_frequency, alarm_duration)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()