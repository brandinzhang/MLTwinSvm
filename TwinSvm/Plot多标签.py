import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler
from TwinSvm import MultiLabelTwinSvm

# 生成具有两个特征的多标签分类数据集，总共有 3 个类别且每个样本平均有 2 个标签
X, Y = make_multilabel_classification(n_samples=100, n_features=2, n_classes=3, n_labels=2,
                                      random_state=1)

# 加点噪声
rng = np.random.RandomState(2)
X += 0.2 * rng.uniform(size=X.shape)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 创建并训练 MultiLabelTwinSvm 模型
model = MultiLabelTwinSvm(kernel='rbf', gamma=0.7)
model.fit(X, Y)

# 创建网格点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点类别
Z_list = []
for sub_model in model.models:
    Z = sub_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z_list.append(Z)

# 绘制决策边界和数据点
n_labels = Y.shape[1]
fig, axes = plt.subplots(1, n_labels, figsize=(15, 5))

for i in range(n_labels):
    Z = Z_list[i]
    y = Y[:, i]
    sub_model = model.models[i]

    # 绘制决策边界
    axes[i].contourf(xx, yy, Z, alpha=0.4)

    # 绘制数据点
    axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=80)

    # 假设 TwinSvm 类有 get_support_vectors 方法
    try:
        sv = sub_model.get_support_vectors()
        # 绘制支持向量
        axes[i].scatter(sv[:, 0], sv[:, 1],
                        s=200, facecolors='none', edgecolors='r', label='Support Vectors')
    except AttributeError:
        pass

    # 设置图形标题和坐标轴标签
    axes[i].set_title(f'Label {i + 1} Classification')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')
    axes[i].legend()

plt.tight_layout()
plt.show()