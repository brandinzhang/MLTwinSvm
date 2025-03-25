import Read
from sklearn.preprocessing import StandardScaler
from twsvmlib import MLTSVM_k as K
from twsvmlib import metrics as M


datasets = ['flags','birds','emotions','yeast',]
paras = [
    {'c1':0.5,'gamma':0.1}, 
    {'c1':0.5,'gamma':0.1},
    {'c1':0.7,'gamma':0.1},
    {'c1':0.9,'gamma':0.1},
]

"""
for dataset, para in zip(datasets, paras):
    X, y = Read.read_train(dataset)
    X_test, y_test = Read.read_test(dataset)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    model = K.kMLTSVM(**para, kernel = 'rbf')
    t1 = time.time()
    model.fit(X, y)
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
    print("\n")
"""

X, y = Read.read_train('flags')
X_test, y_test = Read.read_test('flags')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


model = K.kMLTSVM(c1 = 0.5, gamma = 1.0, kernel = 'rbf')
model.fit(X, y)
y_pred = model.predict(X_test)
y_score = model.score(X_test)
metrics = {
            "Hamming Loss": M.hamming_loss(y_test, y_pred),
            "Ranking Loss": M.ranking_loss(y_test, y_score),
             "One-Error": M.one_error(y_test, y_score),
            "Coverage": M.coverage(y_test, y_score)
    }
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
