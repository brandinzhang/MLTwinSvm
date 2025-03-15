
### cvxopt

`cvxopt` 是一个用于凸优化的 Python 库，可用来解决二次规划问题。二次规划问题的标准形式如下：

$$
\begin{align*}
\min_{x} &\quad \frac{1}{2} x^T P x + q^T x \\
\text{s.t.} &\quad Gx \leq h \\
&\quad Ax = b
\end{align*}
$$
```python
from cvxopt import matrix, solvers
# 定义二次规划问题的参数
P = matrix([[1.0, 0.0], [0.0, 1.0]])  # 半正定矩阵 P
q = matrix([-1.0, -1.0])  # 向量 q
G = matrix([[-1.0, 0.0], [0.0, -1.0]])  # 不等式约束矩阵 G
h = matrix([0.0, 0.0])  # 不等式约束向量 h
A = matrix([[1.0, 1.0]], tc='d')  # 等式约束矩阵 A
b = matrix([1.0])  # 等式约束向量 b
# 求解二次规划问题
sol = solvers.qp(P, q, G, h, A, b)
# 输出结果
print('最优解:')
print(sol['x'])
print('最优值:')
print(sol['primal objective'])

```

多个不等式约束的情况
```py
import numpy as np
from cvxopt import matrix, solvers

# 定义二次规划问题的参数
P = matrix([[1.0, 0.0], [0.0, 1.0]])  # 半正定矩阵 P
q = matrix([-1.0, -1.0])  # 向量 q

# 定义 G 为 numpy 数组
G_np = np.array([[1.0, 0.0], [0.0, 1.0]])
h = matrix([2.0, 2.0])  # 不等式约束的右侧向量

# 处理 0 <= Gx <= h 的约束
# 对于 Gx >= 0，转化为 -Gx <= 0
G_negative_np = -G_np

# 合并两个不等式约束的 numpy 数组
G_combined_np = np.vstack((G_negative_np, G_np))

# 将合并后的 numpy 数组转换为 cvxopt 矩阵
G = matrix(G_combined_np)

# 创建零向量
zeros = matrix([0.0, 0.0])
h_combined = matrix(np.vstack((zeros, h)))

A = matrix([[1.0, 1.0]], tc='d')  # 等式约束矩阵 A
b = matrix([1.0])  # 等式约束向量 b

# 求解二次规划问题
sol = solvers.qp(P, q, G, h_combined, A, b)

# 输出结果
print('最优解:')
print(sol['x'])
print('最优值:')
print(sol['primal objective'])
```


#### 代码注意事项

- 需要注意，`cvxopt` 的 `matrix` 函数默认使用浮点数，因此对于整数矩阵，需要使用 `tc='d'` 明确指定为双精度浮点数。
- cvxopt 的 solvers.qp 函数返回一个字典 sol，'x'：存储问题的最优解向量
- cvxopt接受的是matrix数据类型，nparray需要先转换一下：`cvx_matrix = matrix(np_array,tc=d)`
- 不存在的矩阵对应约束填None

### 生成二分类数据

```py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 生成分类数据
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_classes=2)
"""
n_samples=1000：生成 1000 个样本。
n_features=2：每个样本有 2 个特征。
n_redundant=0：没有冗余特征。
n_informative=2：有 2 个信息特征。
random_state=1：设置随机种子，保证结果可复现。
n_clusters_per_class=1：每个类别有 1 个簇。
n_classes=2：有 2 个类别。
"""


# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建 SVM 模型
svm_model = SVC(kernel='linear')

# 训练模型
svm_model.fit(X_train, y_train)

# 预测测试集
y_pred = svm_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM 模型的准确率: {accuracy:.2f}")
    
```

### 可视化

```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 生成具有两个特征的分类数据集
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_classes=2)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 创建并训练 SVM 模型
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# 创建网格点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))



# 预测网格点类别
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
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
Z 是之前使用 svm_model.predict() 函数对网格点进行类别预测后得到的二维数组，其形状与 xx 和 yy 相同。Z 中的每个元素对应一个网格点的预测类别。
"""



# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=80)

# 绘制支持向量
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='r', label='Support Vectors')

# 设置图形标题和坐标轴标签
plt.title('SVM Classification with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
    
```

### 列合并单位1向量

```py
import numpy as np

# 示例矩阵 A
A = np.array([[1, 2], [3, 4], [5, 6]])

# 示例单位列向量 e
e = np.ones((A.shape[0], 1))

# 使用 numpy.c_ 进行合并
result_c = np.c_[A, e]
```
### 单位矩阵创建

```py
import numpy as np

# 使用 eye 函数创建单位矩阵
size = 5
eye_matrix = np.eye(size)
print("使用 eye 函数创建的单位矩阵：")
print(eye_matrix)

# 使用 identity 函数创建单位矩阵
identity_matrix = np.identity(size)
print("使用 identity 函数创建的单位矩阵：")
print(identity_matrix)
```



