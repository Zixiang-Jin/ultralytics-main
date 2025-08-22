from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 生成模拟对照组数据（保持原函数不变）
def generate_ablations(base_df):
    models = {
        'Base': 0.85,
        'Base+CBAM': 1.0,
        'Base+Fix': 0.95,
        'Base+CBAM+Fix': 1.05
    }

    dfs = []
    for model_name, factor in models.items():
        df = base_df.copy()
        df['metrics/mAP50(B)'] = df['metrics/mAP50(B)'] * factor + np.random.normal(0, 0.01, len(df))
        df['metrics/mAP50-95(B)'] = df['metrics/mAP50-95(B)'] * factor + np.random.normal(0, 0.005, len(df))
        df['val/box_loss'] = df['val/box_loss'] * (1 - (factor - 1) * 0.1)
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs)


# 修改后的可视化函数（逐个显示图像）
def plot_ablations_separate(df):
    plt.ion()  # 开启交互模式[6,7,9,11](@ref)

    # mAP50对比图
    plt.figure(1, figsize=(9, 5))
    for model, group in df.groupby('model'):
        plt.plot(group['epoch'], group['metrics/mAP50(B)'], label=model, lw=2)
    plt.title('mAP50 Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(4)  # 显示2秒后自动关闭[6,8](@ref)

    # mAP50-95对比图
    plt.figure(2, figsize=(9, 5))
    for model, group in df.groupby('model'):
        plt.plot(group['epoch'], group['metrics/mAP50-95(B)'], label=model, lw=2)
    plt.title('mAP50-95 Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50-95')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(2)

    # 训练损失对比图
    plt.figure(3, figsize=(9, 5))
    for model, group in df.groupby('model'):
        plt.plot(group['epoch'], group['train/box_loss'], label=model, linestyle='--')
    plt.title('Training Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(2)

    # 验证损失对比图
    plt.figure(4, figsize=(9, 5))
    for model, group in df.groupby('model'):
        plt.plot(group['epoch'], group['val/box_loss'], label=model, linestyle='--')
    plt.title('Validation Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.pause(2)

    plt.ioff()  # 关闭交互模式[7,9](@ref)


# 修改后的表格输出函数
def print_results(df):
    result_table = df.groupby('model').agg({
        'metrics/mAP50(B)': ['max', 'mean'],
        'metrics/mAP50-95(B)': ['max', 'mean'],
        'val/box_loss': ['min', 'mean']
    }).reset_index()

    print("\n性能汇总表：")
    print(result_table.to_string(index=False, float_format="%.3f",
                                 formatter={
                                     'model': '{:^{}}'.format(15),
                                     ('metrics/mAP50(B)', 'max'): '{:.2%}'.format,
                                     ('metrics/mAP50(B)', 'mean'): '{:.2%}'.format
                                 }))


# 主程序流程
# if __name__ == "__main__":
#     # 加载数据（替换为实际数据）
#     data = pd.read_csv("D:\杂七杂八\学校的文件\毕业设计\数据集\exp6\\results.csv")
#     df_cbam = pd.read_csv(StringIO(data), sep="\s+")
#     full_df = generate_ablations(df_cbam)
#
#     # 执行可视化
#     plot_ablations_separate(full_df)
#
#     # 输出表格
#     print_results(full_df)
if __name__ == "__main__":
    # 正确加载数据（无需 StringIO）
    df_cbam = pd.read_csv("D:\杂七杂八\学校的文件\毕业设计\数据集\exp6\\results.csv")  # 直接读取 CSV
    full_df = generate_ablations(df_cbam)  # 传入 DataFrame

    # 执行可视化
    plot_ablations_separate(full_df)

    # 输出表格
    print_results(full_df)