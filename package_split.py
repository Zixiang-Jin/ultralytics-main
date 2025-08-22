import argparse
import glob
import random
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 自动计算最佳线程数
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))


def run(func, this_iter, desc="Processing"):
    """带进度条的线程池执行器"""
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc))
    return results


def split_dataset_into_train_val_test(
        dataset_dir: Path,
        save_dir: Path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        im_suffix=('jpg', 'png', 'jpeg')
):
    """数据集划分主函数"""
    # 获取所有图片路径
    image_files = []
    for suffix in im_suffix:
        image_files += list((dataset_dir / "Images").glob(f"*.{suffix}"))

    # 随机打乱并划分数据集
    random.shuffle(image_files)
    total = len(image_files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    # 打印统计信息
    print(f"\n{'*' * 30}\n数据集划分结果：")
    print(f"总样本数：{total}")
    print(f"训练集：{len(splits['train'])}")
    print(f"验证集：{len(splits['val'])}")
    print(f"测试集：{len(splits['test'])}\n{'*' * 30}")

    # 创建目标目录结构
    for split_name in splits.keys():
        (save_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (save_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # 多线程处理文件复制
    for split_name, files in splits.items():
        args_list = [(img_path, dataset_dir, save_dir / split_name) for img_path in files]
        run(process_image, args_list, desc=f"处理 {split_name} 集")


def process_image(args):
    """处理单个图片的复制和标注检查"""
    img_path, src_dir, dst_dir = args

    # 获取对应标注文件路径
    label_path = src_dir / "labels" / f"{img_path.stem}.txt"

    # 验证标注文件存在性
    if not label_path.exists():
        print(f"警告：缺失标注文件 {label_path}")
        return

    # 检查标注是否有效
    if not has_objects(label_path):
        return

    # 执行文件复制（使用shutil.copy2保留元数据）
    shutil.copy2(img_path, dst_dir / "images" / img_path.name)
    shutil.copy2(label_path, dst_dir / "labels" / label_path.name)


def has_objects(label_path: Path) -> bool:
    """检查标注文件是否包含有效目标"""
    with open(label_path, 'r', encoding='utf-8') as f:
        return len(f.readlines()) > 0


if __name__ == "__main__":
    # 默认路径配置（适配中文Windows系统）
    DEFAULT_ROOT = Path(r"D:/杂七杂八/学校的文件/毕业设计/数据集/split_CTIR_Dataset").resolve()

    # 命令行参数配置
    parser = argparse.ArgumentParser(description='YOLO数据集划分工具')
    parser.add_argument('--data', type=Path, default=DEFAULT_ROOT, help='数据集根目录')
    parser.add_argument('--save', type=Path, default=DEFAULT_ROOT, help='保存路径')
    parser.add_argument('--suffix', nargs='+', default=['jpg', 'png'], help='支持的图片格式')
    args = parser.parse_args()

    # 路径验证
    required_dirs = ["Images", "labels"]
    for d in required_dirs:
        if not (args.data / d).exists():
            raise FileNotFoundError(f"必需目录缺失: {args.data / d}")

    # 执行数据集划分
    split_dataset_into_train_val_test(
        dataset_dir=args.data,
        save_dir=args.save,
        im_suffix=args.suffix
    )