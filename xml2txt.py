import xml.etree.ElementTree as ET
import os
import random
from pathlib import Path


# 归一化函数
def convert(size, box):
    """将Pascal VOC坐标转换为YOLO格式的归一化坐标"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


# XML转TXT标注文件函数
def convert_annotation(root_path, image_id, classes):
    """转换单个XML标注文件为YOLO格式"""
    root_path = Path(root_path)

    # 输入输出路径
    xml_file = root_path / "Annotations" / f"{image_id}.xml"
    label_dir = root_path / "labels"
    txt_file = label_dir / f"{image_id}.txt"

    # 确保目录存在
    label_dir.mkdir(parents=True, exist_ok=True)

    with open(xml_file, 'r', encoding='utf-8') as in_file, \
            open(txt_file, 'w', encoding='utf-8') as out_file:

        tree = ET.parse(in_file)
        root = tree.getroot()

        # 获取图像尺寸（带容错机制）
        size = root.find("size")
        if size is not None:
            w = int(size.find("width").text)
            h = int(size.find("height").text)
        else:
            # 从图片文件获取尺寸（需要opencv-python）
            img_path = root_path / "images" / f"{image_id}.jpg"
            try:
                import cv2
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
            except Exception as e:
                print(f"错误：无法获取 {image_id} 的尺寸: {e}")
                return

        # 处理每个标注对象
        for obj in root.iter("object"):
            # 处理difficult标记
            difficult = obj.find("difficult")
            difficult = int(difficult.text) if difficult is not None else 0

            # 获取类别
            cls_name = obj.find("name").text

            # 过滤类别和difficult标记
            if cls_name not in classes or difficult == 1:
                continue

            # 获取边界框坐标
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)

            # 转换坐标格式
            yolo_coords = convert((w, h), (xmin, xmax, ymin, ymax))

            # 写入TXT文件
            cls_id = classes.index(cls_name)
            out_file.write(f"{cls_id} {' '.join(f'{coord:.6f}' for coord in yolo_coords)}\n")


def create_image_sets(root_path, train_ratio=0.8, val_ratio=0.1):
    """自动创建数据集划分文件"""
    root_path = Path(root_path)
    annotations = [f.stem for f in (root_path / "Annotations").glob("*.xml")]
    random.shuffle(annotations)

    total = len(annotations)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": annotations[:train_end],
        "val": annotations[train_end:val_end],
        "test": annotations[val_end:]
    }

    # 写入划分文件
    (root_path / "ImageSets").mkdir(exist_ok=True)
    for split_name, ids in splits.items():
        with open(root_path / "ImageSets" / f"{split_name}.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(ids))


if __name__ == "__main__":
    # 配置参数
    CLASSES = ["Car", "Pedestrian", "Cyclist", "Bus", "Truck"]
    DATASET_ROOT = r"D:/杂七杂八/学校的文件/毕业设计/数据集/split_CTIR_Dataset"

    # 初始化路径
    dataset_path = Path(DATASET_ROOT).resolve()
    (dataset_path / "labels").mkdir(exist_ok=True)  # 创建labels目录

    # 步骤1：自动创建数据集划分
    create_image_sets(dataset_path)
    print("数据集划分文件已创建")

    # 步骤2：转换所有标注文件
    for split in ["train", "val", "test"]:
        # 读取划分文件
        split_file = dataset_path / "ImageSets" / f"{split}.txt"
        with open(split_file, 'r', encoding='utf-8') as f:
            image_ids = [line.strip() for line in f]

        # 生成路径列表文件
        list_file = dataset_path / f"{split}.txt"
        with open(list_file, 'w', encoding='utf-8') as f:
            for img_id in image_ids:
                img_path = (dataset_path / "images" / f"{img_id}.jpg").as_posix()
                f.write(f"{img_path}\n")

                # 转换标注文件
                convert_annotation(dataset_path, img_id, CLASSES)

        print(f"{split}集处理完成，共转换{len(image_ids)}个标注文件")

    print("所有处理已完成！")
    print(f"数据集根目录：{dataset_path}")