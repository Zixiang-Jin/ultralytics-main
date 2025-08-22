# import cv2
# import numpy as np
# from ultralytics import YOLO
#
#
# class DualModelDetector:
#     def __init__(self, model1_path='best_visible.pt', model2_path='best_thermal.pt'):
#         self.model_vis = YOLO(model1_path)
#         self.model_ir = YOLO(model2_path)
#
#     def _calc_brightness(self, img):
#         """计算图像平均亮度（0-255）"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return np.mean(gray)
#
#     def detect(self, img_path):
#         # 读取图像并计算亮度
#         img = cv2.imread(img_path)
#         brightness = self._calc_brightness(img)
#
#         # 动态权重分配（Sigmoid函数）
#         vis_weight = 1 / (1 + np.exp(-0.1 * (brightness - 50)))  # [9](@ref)
#         ir_weight = 1 - vis_weight
#
#         # 双模型推理
#         res_vis = self.model_vis(img)[0].boxes.data.cpu().numpy()  # [xyxy, conf, cls]
#         res_ir = self.model_ir(img)[0].boxes.data.cpu().numpy()
#
#         # 置信度加权融合
#         res_vis[:, 4] *= vis_weight
#         res_ir[:, 4] *= ir_weight
#
#         # 合并结果并NMS处理
#         fused = np.concatenate([res_vis, res_ir], axis=0)
#         final = self._nms(fused, iou_thres=0.5, conf_thres=0.25)
#
#         return final
#
#     def _nms(self, boxes, iou_thres=0.5, conf_thres=0.25):
#         """非极大值抑制实现"""
#         # 按置信度排序
#         boxes = boxes[boxes[:, 4].argsort()[::-1]]
#         keep = []
#
#         while boxes.shape[0]:
#             # 取当前最高置信度框
#             current = boxes[0]
#             keep.append(current)
#
#             # 计算IoU
#             ious = []
#             for box in boxes[1:]:
#                 x1 = max(current[0], box[0])
#                 y1 = max(current[1], box[1])
#                 x2 = min(current[2], box[2])
#                 y2 = min(current[3], box[3])
#
#                 inter = max(0, x2 - x1) * max(0, y2 - y1)
#                 area1 = (current[2] - current[0]) * (current[3] - current[1])
#                 area2 = (box[2] - box[0]) * (box[3] - box[1])
#                 iou = inter / (area1 + area2 - inter)
#                 ious.append(iou)
#
#             # 过滤重叠框
#             boxes = boxes[1:][np.array(ious) < iou_thres]
#
#         return np.array(keep)
#
#
# # 使用示例
# detector = DualModelDetector()
# results = detector.detect('test.jpg')
#
# # 可视化输出
# img = cv2.imread('test.jpg')
# for box in results:
#     x1, y1, x2, y2, conf, cls = box.astype(int)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.imwrite('result.jpg', img)

# import cv2
# import numpy as np
# from ultralytics import YOLO
#
#
# class DualModelDetector:
#     def __init__(self, model1_path='runs/train/exp01/weights/best.pt', model2_path='runs/train/exp02/weights/best.pt'):
#         self.model_vis = YOLO(model1_path)  # 可见光模型
#         self.model_ir = YOLO(model2_path)  # 红外模型
#
#     def _calc_brightness(self, img):
#         """计算可见光图像亮度（0-255）"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return np.mean(gray)
#
#     def detect(self, vis_path, ir_path):
#         """
#         双模态融合检测
#         :param vis_path: 可见光图像路径
#         :param ir_path: 红外图像路径
#         :return: 融合后的检测结果[n,6] (xyxy, conf, cls)
#         """
#         # 读取双模态图像
#         img_vis = cv2.imread(vis_path)
#         img_ir = cv2.imread(ir_path)
#
#         # 计算可见光图像亮度
#         brightness = self._calc_brightness(img_vis)
#
#         # 动态权重分配（基于Sigmoid函数）
#         vis_weight = 1 / (1 + np.exp(-0.1 * (brightness - 50)))
#         ir_weight = 1 - vis_weight
#
#         # 双模型并行推理
#         res_vis = self.model_vis(img_vis)[0].boxes.data.cpu().numpy()  # 可见光结果
#         res_ir = self.model_ir(img_ir)[0].boxes.data.cpu().numpy()  # 红外结果
#
#         # 置信度加权融合
#         res_vis[:, 4] *= vis_weight  # 可见光结果置信度加权
#         res_ir[:, 4] *= ir_weight  # 红外结果置信度加权
#
#         # 合并结果并NMS处理
#         fused = np.concatenate([res_vis, res_ir], axis=0)
#         final = self._nms(fused, iou_thres=0.5, conf_thres=0.25)
#
#         return final
#
#     def _nms(self, boxes, iou_thres=0.5, conf_thres=0.25):
#         """非极大值抑制"""
#         keep = []
#         # 按置信度降序排序
#         boxes = boxes[boxes[:, 4].argsort()[::-1]]
#
#         while len(boxes) > 0:
#             # 取当前最高置信度框
#             current = boxes[0]
#             if current[4] < conf_thres:
#                 break
#             keep.append(current)
#
#             # 计算与剩余框的IoU
#             if len(boxes) == 1:
#                 break
#             current_area = (current[2] - current[0]) * (current[3] - current[1])
#             boxes = boxes[1:]
#
#             # 计算交并比
#             x1 = np.maximum(boxes[:, 0], current[0])
#             y1 = np.maximum(boxes[:, 1], current[1])
#             x2 = np.minimum(boxes[:, 2], current[2])
#             y2 = np.minimum(boxes[:, 3], current[3])
#             intersections = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
#
#             unions = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + \
#                      current_area - intersections
#             ious = intersections / unions
#
#             # 保留IoU低于阈值的框
#             boxes = boxes[ious <= iou_thres]
#
#         return np.array(keep)
#
#
# # 使用示例
# if __name__ == "__main__":
#     detector = DualModelDetector()
#
#     # 输入成对的可见光和红外图像
#     vis_img = "datasets/roadspic/images/train/010001.jpg"
#     ir_img = r"D:/杂七杂八/学校的文件/毕业设计/数据集/LLVIP/infrared/train/010001.jpg"
#
#     # 执行融合检测
#     results = detector.detect(vis_img, ir_img)
#
#     # 可视化结果（默认在可见光图像上绘制）
#     img = cv2.imread(vis_img)
#     for box in results:
#         x1, y1, x2, y2, conf, cls = map(int, box[:6])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     cv2.imwrite("fusion_result.jpg", img)

import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


class DualModelDetector:
    def __init__(self, model1_path='runs/train/exp01/weights/best.pt',
                 model2_path='runs/train/exp02/weights/best.pt'):
        self.model_vis = YOLO(model1_path)  # 可见光模型
        self.model_ir = YOLO(model2_path)  # 红外模型

    def _calc_brightness(self, img):
        """计算可见光图像亮度（0-255）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def detect(self, vis_path, ir_path):
        """
        双模态融合检测
        :param vis_path: 可见光图像路径
        :param ir_path: 红外图像路径
        :return: 融合后的检测结果[n,6] (xyxy, conf, cls)
        """
        # 读取双模态图像
        img_vis = cv2.imread(vis_path)
        img_ir = cv2.imread(ir_path)

        # 计算可见光图像亮度
        brightness = self._calc_brightness(img_vis)

        # 动态权重分配（基于Sigmoid函数）
        vis_weight = 1 / (1 + np.exp(-0.1 * (brightness - 50)))
        ir_weight = 1 - vis_weight

        # 双模型并行推理
        res_vis = self.model_vis(img_vis)[0].boxes.data.cpu().numpy()  # 可见光结果
        res_ir = self.model_ir(img_ir)[0].boxes.data.cpu().numpy()  # 红外结果

        # 置信度加权融合
        res_vis[:, 4] *= vis_weight
        res_ir[:, 4] *= ir_weight

        # 合并结果并NMS处理
        fused = np.concatenate([res_vis, res_ir], axis=0)
        final = self._nms(fused, iou_thres=0.5, conf_thres=0.25)

        return final

    def _nms(self, boxes, iou_thres=0.5, conf_thres=0.25):
        """非极大值抑制"""
        keep = []
        boxes = boxes[boxes[:, 4].argsort()[::-1]]  # 按置信度降序排序

        while len(boxes) > 0:
            current = boxes[0]
            if current[4] < conf_thres:
                break
            keep.append(current)

            if len(boxes) == 1:
                break

            # 计算IoU
            x1 = np.maximum(boxes[1:, 0], current[0])
            y1 = np.maximum(boxes[1:, 1], current[1])
            x2 = np.minimum(boxes[1:, 2], current[2])
            y2 = np.minimum(boxes[1:, 3], current[3])

            intersections = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            unions = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1]) + \
                     (current[2] - current[0]) * (current[3] - current[1]) - intersections
            ious = intersections / unions

            boxes = boxes[1:][ious <= iou_thres]

        return np.array(keep)


def process_folder(vis_folder, ir_folder, output_folder, detector):
    """批量处理文件夹"""
    os.makedirs(output_folder, exist_ok=True)

    # 获取可见光文件列表
    vis_files = sorted([f for f in os.listdir(vis_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    pbar = tqdm(vis_files, desc="Processing images")
    for vis_file in pbar:
        # 构建对应路径
        vis_path = os.path.join(vis_folder, vis_file)
        ir_path = os.path.join(ir_folder, vis_file)  # 假设文件名相同

        if not os.path.exists(ir_path):
            continue

        # 执行检测
        results = detector.detect(vis_path, ir_path)

        # 可视化并保存
        img = cv2.imread(vis_path)
        for box in results:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存结果
        output_path = os.path.join(output_folder, vis_file)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    # 初始化检测器
    detector = DualModelDetector()

    # 设置路径
    VIS_FOLDER = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\\visible\\test"
    IR_FOLDER = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\infrared\\val"
    OUTPUT_FOLDER = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\output_result"

    # 执行批量处理
    process_folder(VIS_FOLDER, IR_FOLDER, OUTPUT_FOLDER, detector)