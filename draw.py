import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（需先处理csv格式）
df = pd.read_csv('D:\杂七杂八\学校的文件\毕业设计\数据集\exp6\\results.csv')

plt.figure(figsize=(15, 20))
sns.set_style("whitegrid")
plt.suptitle("Training Metrics Analysis", y=0.92)

# 损失函数分析
plt.subplot(3,1,1)
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.plot(df['epoch'], df['train/dfl_loss'], '--', label='Train DFL Loss')
plt.plot(df['epoch'], df['val/dfl_loss'], '--', label='Val DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Box & DFL Loss Progression')

# 关键指标分析
plt.subplot(3,1,2)
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.plot(df['epoch'], df['metrics/precision(B)'], '--', label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], '--', label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Key Detection Metrics')

# 学习率动态
plt.subplot(3,1,3)
plt.plot(df['epoch'], df['lr/pg0'], label='PG0 Learning Rate')
plt.plot(df['epoch'], df['lr/pg1'], label='PG1 Learning Rate')
plt.plot(df['epoch'], df['lr/pg2'], label='PG2 Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.show()