import numpy as np
import Read
import pandas as pd
from sklearn.preprocessing import StandardScaler
from twsvmlib import MLTSVM_ova as O
from twsvmlib import metrics as M
import time

datasets = ['flags']
paras = [
    {'c1':0.1,'gamma':0.3,'kernel':'rbf'}, 

]
for dataset, para in zip(datasets, paras):
    X, y = Read.read_train(dataset)
    X_test, y_test = Read.read_test(dataset)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    model = O.ovaMLTSVM(**para)
    
    model.fit(X, y)
    y_pred = model.predict(X_test)

    



    """
    acu = np.mean(y_pred == y_test, axis=0)
    spar = np.mean(y_test, axis=0)
    print(f"{dataset}每个标签的准确率: {acu}")
    print(f"{dataset}每个标签的稀疏度：{spar}")
    """

    """
    # 保存y_pred为csv文件到指定路径
    Read.save_pred(y_pred, dataset)
    print(f"{dataset}的预测结果已保存")
    """
    
    
    t1 = time.time()
    t2 = time.time()
    y_pred = model.predict(X_test)
    y_score = model.score(X_test)

    print(f"ktsvm在数据集: {dataset}训练用时: {t2-t1:.2f}s")
    metrics = {
            "Hamming Loss": M.hamming_loss(y_test, y_pred),
            "Ranking Loss": M.ranking_loss(y_test, y_score),
             "One-Error": M.one_error(y_test, y_score),
            "Coverage": M.coverage(y_test, y_score)
    }
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    

        

