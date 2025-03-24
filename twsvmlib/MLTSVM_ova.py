import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from . import TwinSvm as T


class ovaMLTSVM(BaseEstimator, ClassifierMixin):
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
            model = T.TwinSvm(c1=self.c1, c2=self.c2, Epsi1=self.Epsi1, Epsi2=self.Epsi2,
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