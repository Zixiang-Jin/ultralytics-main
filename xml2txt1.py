import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(xml_folder, output_folder, class_ids):
    """
    批量将 Pascal VOC XML 标注文件转换为 YOLO 格式的 TXT 文件

    参数:
        xml_folder (str): XML 文件所在文件夹路径
        output_folder (str): 输出 TXT 文件保存路径
        class_ids (dict): 类别名称到 ID 的映射字典，例如 {'person': 0}
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历所有 XML 文件
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        # 解析 XML
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 准备输出文件
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, 'w') as f:
            # 遍历每个检测目标
            for obj in root.iter('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in class_ids:
                    continue  # 跳过未定义类别

                # 获取边界框坐标
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 转换为 YOLO 格式（归一化中心坐标和宽高）
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                # 写入文件
                f.write(f"{class_ids[cls_name]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    # 使用示例
    XML_FOLDER = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\Annotations"  # 替换为你的 XML 文件夹路径
    OUTPUT_FOLDER = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\Annotations2"  # 替换为输出文件夹路径
    CLASS_DICT = {"person": 0}  # 根据你的数据集类别修改

    convert_voc_to_yolo(XML_FOLDER, OUTPUT_FOLDER, CLASS_DICT)
    print(f"转换完成！文件已保存至 {OUTPUT_FOLDER}")