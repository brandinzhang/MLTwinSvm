import KernelMatrix as km
import numpy as np




# 创建一个二维矩阵 X
X = np.array([[1, 2],
              [4, 5],
              [7, 8]])

# 创建一个一维向量 y
y = np.array([[1, 2, 3],
              [4, 5, 6]])



print("矩阵 X:")
print(X)
print("向量 y:")
print(y)
print("线性核矩阵:")
print(km.linear_kernel(X, y))
print("RBF核矩阵:")
print(km.rbf_kernel(X, y, gamma=0.1))


