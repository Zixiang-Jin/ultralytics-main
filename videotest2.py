import cv2
from ultralytics import YOLO
import winsound
from threading import Thread

model = YOLO("runs/detect/train3/weights/best.pt")

# 初始化摄像头并设置分辨率
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率以提升CPU速度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 报警参数
target_class = "car"
alarm_threshold = 0.7


def play_alarm():
    """播放报警声音（Windows系统）"""
    try:
        winsound.MessageBeep()
    except:
        print("蜂鸣器不可用，请检查系统权限或改用其他音频库")


# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, device='cpu')

    # 检测结果处理
    alarm_triggered = False
    for result in results:
        # 绘制检测框到画面
        annotated_frame = result.plot()
        cv2.imshow("Frame", annotated_frame)

        # 遍历每个检测框
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()  # 置信度值
            cls_name = model.names[cls_id].lower()  # 统一转为小写

            # 触发条件判断
            if cls_name == target_class.lower() and conf >= alarm_threshold:
                print(f"触发报警: {cls_name} (置信度: {conf:.2f})")
                alarm_triggered = True
                break  # 发现目标即跳出当前循环

    # 避免重复触发：每帧只报警一次
    if alarm_triggered:
        Thread(target=play_alarm).start()  # 非阻塞播放

    # 退出检测（ESC键）
    if cv2.waitKey(1) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()