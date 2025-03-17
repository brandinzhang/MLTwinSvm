import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import KernelMatrix as km
import TsvmPlane1
import TsvmPlane2


class MultiLabelTwinSvm(BaseEstimator, ClassifierMixin):
    def __init__(self, c1=1, c2=1, Epsi1=0.01, Epsi2=0.01, kernel='linear', degree=2, gamma=1.0, r=0):
        self.c1 = c1
        self.c2 = c2
        self.Epsi1 = Epsi1
        self.Epsi2 = Epsi2
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.models = []

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        for i in range(n_labels):
            y = Y[:, i]
            model = TwinSvm(c1=self.c1, c2=self.c2, Epsi1=self.Epsi1, Epsi2=self.Epsi2,
                            kernel=self.kernel, degree=self.degree, gamma=self.gamma, r=self.r)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions).T

    def score(self, X, Y, sample_weight=None):
        from sklearn.metrics import hamming_loss
        Y_pred = self.predict(X)
        return 1 - hamming_loss(Y, Y_pred)



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
        self.u1,self.b1,self.alpha = TsvmPlane1.solve(R,S,self.c1,self.Epsi1)
        self.u2,self.b2,self.beta  = TsvmPlane2.solve(R,S,self.c2,self.Epsi2)
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
    def get_support_vectors(self):
        """
        获取支持向量。
        :return: 支持向量矩阵
        """
        mask1 = self.alpha > 1e-5
        mask1 = mask1.flatten()
        mask2 = self.beta > 1e-5
        mask2 = mask2.flatten()
        sv1 = self.A[mask2]
        sv2 = self.B[mask1]
        return np.vstack((sv1, sv2))

def test():
    # 生成多标签模拟数据
    X, Y = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, n_labels=2, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 创建 MultiLabelTwinSvm 分类器实例
    clf = MultiLabelTwinSvm()

    # 训练分类器
    clf.fit(X_train, Y_train)

    # 预测测试集
    Y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = clf.score(X_test, Y_test)

    print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    test()


    
