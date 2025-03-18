import os
import pandas as pd
import numpy as np

# 切换到项目根目录
os.chdir('/Users/asina6y6/Desktop/test/TwinSvm')

# 定义每个数据集的标签列索引范围（这里需要根据实际情况修改）
label_cols = {
    'birds': slice(-19, None),  # 假设 birds 数据集最后 19 列是标签列
    'emotions': slice(-6, None),  # 假设 emotions 数据集最后 6 列是标签列
    'flags': slice(-7, None),  # 假设 flags 数据集最后 7 列是标签列
    'yeast': slice(-14, None)  # 假设 yeast 数据集最后 14 列是标签列   
}

def read_train(name):
    # 读取数据集
    path = 'dataset/' + name + '/' + name + '-train.csv'
    data = pd.read_csv(path, header=None)
    X = data.iloc[:, :label_cols[name].start].values
    Y = data.iloc[:, label_cols[name]].values
    return X,Y

def read_test(name):
    # 读取数据集
    path = 'dataset/' + name + '/' + name + '-test.csv'
    # 不读取第一行
    data = pd.read_csv(path, header = None, skiprows = 1)
    X = data.iloc[:, :label_cols[name].start].values
    Y = data.iloc[:, label_cols[name]].values
    return X,Y

"""
# 读取数据集
X,y = read_test('emotions')
print(X)
print(y)
"""







