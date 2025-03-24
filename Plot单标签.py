import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from twsvmlib import TwinSvm as T
# 生成具有两个特征的分类数据集
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_classes=2)

# 加点噪声
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)



# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 创建并训练 TSVM 模型
model = T.TwinSvm(kernel='rbf', gamma=0.7)
model.fit(X,y)


# 创建网格点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))



# 预测网格点类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
"""
ravel()是平铺为一维数组
np.c_[xx.ravel(), yy.ravel()] 把 xx.ravel() 和 yy.ravel() 按列连接起来，形成一个二维数组，这个二维数组的每一行代表一个网格点的坐标（即特征）。
"""


# 绘制决策边界
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
"""
xx 和 yy 是通过 np.meshgrid() 函数生成的二维数组，它们分别表示网格点的 x 坐标和 y 坐标。
Z 是之前使用 model.predict() 函数对网格点进行类别预测后得到的二维数组，其形状与 xx 和 yy 相同。Z 中的每个元素对应一个网格点的预测类别。
"""



# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=80)

sv = model.get_support_vectors()

# 绘制支持向量
plt.scatter(sv[:, 0], sv[:, 1],
            s=200, facecolors='none', edgecolors='r', label='Support Vectors')

# 设置图形标题和坐标轴标签
plt.title('TSVM Classification with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

