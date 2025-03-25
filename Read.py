import pandas as pd





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
def save_pred(y_pred, name):
    # 保存预测结果
    path = 'dataset/' + name + '/' + name + '-pred.csv'
    pd.DataFrame(y_pred).to_csv(path, index=False, header=False)


"""
# 读取数据集
X,y = read_test('birds')
print(X)
print(y)
"""









