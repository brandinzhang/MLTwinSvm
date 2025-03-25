import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from . import KernelMatrix as km
from . import TsvmPlane1
from . import TsvmPlane2

def sigmod(x):
    return 1/(1+np.exp(-x))

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
            xc = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            xc = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
        elif self.kernel == 'rbf':
            xc = km.rbf_kernel(X, C.T, self.gamma)
        else:
            xc = km.linear_kernel(X, C.T)

        # 计算到两个平面的绝对值距离
        distance1 = np.abs(xc @ self.u1 + self.b1)
        distance2 = np.abs(xc @ self.u2 + self.b2)
        self.delta = distance2
        # 根据距离选择类别，距离小的类别为预测结果
        predictions = (distance1 < distance2).astype(int)

        # 返回的是一维向量
        return predictions
    def get_delta(self):
        return self.delta
        

    def get_score(self, X):
        """
        返回输入数据的分数
        取-distance作为最终分数
        """
        # 首先将 A 和 B 合并成 C
        C = np.vstack((self.A, self.B))
        if self.kernel == 'linear':
            xc = km.linear_kernel(X, C.T)
        elif self.kernel == 'poly':
            xc = km.poly_kernel(X, C.T, self.gamma, self.r, self.degree)
            
        elif self.kernel == 'rbf':
            # X: t*n C: m*n u1,u2:(m,)
            xc = km.rbf_kernel(X, C.T, self.gamma) #t*m
            cc = km.rbf_kernel(C,C.T,self.gamma)  #m*m
            u1xc = km.rbf_kernel(X,C.T,self.gamma)@self.u1 #(t,)
            u2xc = km.rbf_kernel(X,C.T,self.gamma)@self.u2 #(t,)
            u1ccu2 = self.u1@cc@self.u2  #1*1
            u1ccu1 = self.u1@cc@self.u1  #1*1
            u2ccu2 = self.u2@cc@self.u2  #1*1
        else:
            xc = km.linear_kernel(X, C.T)

        d1 = np.abs(xc @ self.u1 + self.b1)   #(t,)
        d2 = np.abs(xc @ self.u2 + self.b2)   #(t,)

        temp1 = (u1xc+self.b1)/np.sqrt(u1ccu1)  #(t,)
        temp2 = (u2xc+self.b2)/np.sqrt(u2ccu2)  #(t,)
        temp3 = 2*u1ccu2/(u1ccu1*u2ccu2)        #标量
        
        num1 = np.abs(temp1+temp2)  #(t,)
        den1 = np.sqrt(2+temp3)     #标量
        D_pos = num1/den1           #(t,)

        num2 = np.abs(temp1-temp2)  #(t,)
        den2 = np.sqrt(2-temp3)     #标量
        D_neg = num2/den2           #(t,)

        f = np.zeros(d1.shape)       #(t,)
        D = np.maximum(D_pos,D_neg)  #(t,)
        d = np.minimum(D_pos,D_neg)  #(t,)
        temp = d**2/(D+1e-6)         #(t,)

        idx1 = d1<d2
        idx2 = d1>d2
        f[idx1] = temp[idx1]
        f[idx2] = -temp[idx2]

        return sigmod(f)        #(t,)



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
        idx1 = self.alpha > 1e-5
        idx1 = idx1.flatten()
        idx2 = self.beta > 1e-5
        idx2 = idx2.flatten()
        sv1 = self.A[idx2]
        sv2 = self.B[idx1]
        return np.vstack((sv1, sv2))

class pMLTSVM(BaseEstimator, ClassifierMixin):
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
            # 或得对当下标签预测的一维向量
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions).T

    def score(self, X):
        scores = []
        for model in self.models:
            # 或得对当下标签预测的一维向量
            pred = model.get_score(X)
            scores.append(pred)
        return np.array(scores).T
    
    def get_delta_k(self):
        deltas = []
        for model in self.models:
            # 或得对当下标签预测的一维向量
            delta = model.get_delta()
            deltas.append(delta)
        return np.array(deltas)