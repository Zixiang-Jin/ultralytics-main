import os
import shutil


def organize_labels(image_dir, label_source, label_target, img_extensions=None):
    """
    根据图片目录整理对应的标签文件到目标目录
    :param image_dir: 图片所在目录路径
    :param label_source: 原始标签文件目录路径
    :param label_target: 目标标签文件目录路径
    :param img_extensions: 支持的图片扩展名集合
    """
    if img_extensions is None:
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # 确保目标目录存在
    os.makedirs(label_target, exist_ok=True)

    # 统计处理结果
    processed = 0
    missing = 0

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)

        # 跳过目录和非图片文件
        if not os.path.isfile(img_path):
            continue
        if os.path.splitext(img_file)[1].lower() not in img_extensions:
            continue

        # 构造标签文件名
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        src_label = os.path.join(label_source, label_file)
        dst_label = os.path.join(label_target, label_file)

        # 复制标签文件
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            processed += 1
        else:
            print(f"Warning: Missing label for {img_file}")
            missing += 1

    print(f"完成！成功处理 {processed} 个标签，缺失 {missing} 个标签")


if __name__ == "__main__":
    # 配置路径（请根据实际情况修改）
    TRAIN_IMAGES = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\\visible\\train"
    TEST_IMAGES = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\\visible\\test"
    SOURCE_LABELS = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\Annotations1"
    TRAIN_LABELS = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\\visible_label\\train"
    TEST_LABELS = "D:\杂七杂八\学校的文件\毕业设计\数据集\LLVIP\\visible_label\\test"

    # 处理训练集
    print("正在处理训练集...")
    organize_labels(TRAIN_IMAGES, SOURCE_LABELS, TRAIN_LABELS)

    # 处理测试集
    print("\n正在处理测试集...")
    organize_labels(TEST_IMAGES, SOURCE_LABELS, TEST_LABELS)