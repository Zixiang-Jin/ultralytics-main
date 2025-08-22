# import cv2
#
# def find_ir_camera():
#     for index in range(1,10):
#         cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 添加CAP_DSHOW参数
#         if cap.isOpened():
#             print(f"IR摄像头索引号: {index}")
#             cap.release()
#             return index
#         cap.release()
#     return -1
#
# ir_index = find_ir_camera()
# if ir_index != -1:
#     print(f"请将代码中的摄像头ID改为 {ir_index}")
# else:
#     print("未检测到可用摄像头")


# import cv2
#
# # 尝试不同设备ID（0或1）
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 部分机型红外摄像头ID为1
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("摄像头访问失败，请检查设备ID或驱动")
#         break
#
#     # 红外摄像头通常输出灰度图像
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('IR Camera', gray)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2

def list_ports():
    """
    Test the available camera ports and return them as a list
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            camera.release()
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} found but does not read images")
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports

available_ports, working_ports = list_ports()