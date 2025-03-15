import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import KernelMatrix as km
import TsvmPlane1
import TsvmPlane2


class TwinSvm(BaseEstimator, ClassifierMixin):
    def __init__(self,c1=1,c2=1,Epsi1=0.01,Epsi2=0.01,kernel='linear',degree=2,gamma=1.0,r=0):
        """
        初始化 TwinSvm 分类器的参数。
        :param c1: 第一个平面的惩罚参数
        :param c2: 第二个平面的惩罚参数
        :param Epsi1: 第一个平面的正则化系数
        :param Epsi2: 第二个平面的正则化系数
        :param kernel: 核函数类型，可选 'linear', 'poly', 'rbf'
        :param degree: 多项式核的阶数
        :param gamma: 核函数的参数
        :param r: 多项式核的偏移量
        """
        self.c1 = c1
        self.c2 = c2
        self.Epsi1 = Epsi1
        self.Epsi2 = Epsi2
        self.u1 = None
        self.b1 = None
        self.u2 = None
        self.b2 = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
    
    def fit(self,X,Y):
        """
        训练 TwinSvm 分类器。
        :param X: 特征矩阵
        :param y: 标签向量
        :return: 训练好的分类器实例
        """
        Data = sorted(zip(Y,X), key=lambda pair: pair[0], reverse = True)
        
        A=np.array([np.array(x) for y,x in Data if (y==1)])
        B=np.array([np.array(x) for y,x in Data if (y==0)])
        C = np.vstack((A,B))
        m1 = A.shape[0]
        m2 = B.shape[0]
        e1 = np.ones((m1,1))
        e2 = np.ones((m2,1))
        if self.kernel == 'linear':
            K1 = km.linear_kernel(A,C.T)
            K2 = km.linear_kernel(B,C.T)
        elif self.kernel == 'poly':
            K1 = km.poly_kernel(A,C.T,self.gamma,self.r,self.degree)
            K2 = km.poly_kernel(B,C.T,self.gamma,self.r,self.degree)
        elif self.kernel == 'rbf':
            K1 = km.rbf_kernel(A,C.T,self.gamma)
            K2 = km.rbf_kernel(B,C.T,self.gamma)
        else:
            K1 = km.linear_kernel(A,C.T)
            K2 = km.linear_kernel(B,C.T)
        S = np.c_[K1,e1]
        R = np.c_[K2,e2]
        self.u1,self.b1 = TsvmPlane1.solve(R,S,self.c1,self.Epsi1)
        self.u2,self.b2 = TsvmPlane2.solve(R,S,self.c2,self.Epsi2)
        self.A = A
        self.B = B
        return self
    
    def predict(self, X):
        """
        对输入数据进行预测。
        :param X: 待预测的特征矩阵
        :return: 预测的标签向量
        """
        # 首先将 A 和 B 合并成 C
        C = np.vstack((self.A, self.B))
        if self.kernel == 'linear':
            K_x_C = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            K_x_C = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            K_x_C = km.rbf_kernel(X, C.T, self.gamma)
        else:
            K_x_C = km.linear_kernel(X, C.T)

        # 计算到两个平面的绝对值距离
        distance1 = np.abs(K_x_C @ self.u1 + self.b1)
        distance2 = np.abs(K_x_C @ self.u2 + self.b2)

        # 根据距离选择类别，距离小的类别为预测结果
        predictions = (distance1 < distance2).astype(int)
        return predictions

    def score(self, X, y, sample_weight=None):
        """
        计算分类器的准确率。
        :param X: 特征矩阵
        :param y: 真实标签向量
        :param sample_weight: 样本权重
        :return: 准确率
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
    def set_params(self, **params):
        """
        设置估计器的参数。
        :param params: 包含参数名和参数值的字典
        :return: 分类器实例
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param} for estimator {self.__class__.__name__}.")
        return self


def test():
    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 调整参数名以匹配 TwinSvm 类的 __init__ 方法
    params1 = {'c1': 1, 'c2': 1, 'Epsi1': 0.1, 'Epsi2': 0.1, 'kernel': 'linear', 'gamma': 1.0}
    params2 = {'c1': 2, 'c2': 1, 'Epsi1': 0.01, 'Epsi2': 0.01, 'kernel': 'rbf', 'gamma': 1.0}
    params3 = {'c1': 1, 'c2': 1, 'Epsi1': 0.01, 'Epsi2': 0.01, 'kernel': 'rbf', 'gamma': 0.3}

    # 创建 TwinSvm 分类器实例
    clf1 = TwinSvm()
    clf2 = TwinSvm()
    clf3 = TwinSvm()

    # 传入参数
    clf1.set_params(**params1)
    clf2.set_params(**params2)
    clf3.set_params(**params3)

    # 训练分类器
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    # 预测测试集
    y_pred1 = clf1.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    y_pred3 = clf3.predict(X_test)

    # 计算准确率
    accuracy1 = clf1.score(X_test, y_test)
    accuracy2 = clf2.score(X_test, y_test)
    accuracy3 = clf3.score(X_test, y_test)

    print(f"Accuracy with params1: {accuracy1:.2f}")
    print(f"Accuracy with params2: {accuracy2:.2f}")
    print(f"Accuracy with params3: {accuracy3:.2f}")
    
if __name__ == '__main__':
    test()