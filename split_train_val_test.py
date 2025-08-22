
import os
import random

ROOT = os.path.abspath("D:/杂七杂八/学校的文件/毕业设计/数据集/split_CTIR_Dataset/")  # IR 数据集

trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = os.path.join(ROOT, "Annotations")  # 标注数据集文件夹
txtsavepath = os.path.join(ROOT, "ImageSets") # 数据集索引文件夹
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
# 获取该路径下所有文件的名称，存放在list中
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 修改文件创建部分
file_paths = {
    "trainval": os.path.join(txtsavepath, "trainval.txt"),
    "train": os.path.join(txtsavepath, "train.txt"),
    "test": os.path.join(txtsavepath, "test.txt"),
    "val": os.path.join(txtsavepath, "val.txt")
}

with open(file_paths["trainval"], 'w', encoding='utf-8') as ftrainval, \
     open(file_paths["train"], 'w', encoding='utf-8') as ftrain, \
     open(file_paths["test"], 'w', encoding='utf-8') as ftest, \
     open(file_paths["val"], 'w', encoding='utf-8') as fval:

    for i in list:
        name = total_xml[i][:-4] + "\n"
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

print(xmlfilepath)
print(txtsavepath)