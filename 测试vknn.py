import Read
from twsvmlib import MLTSVM_vknn as VK
from twsvmlib import metrics as M






X,y = Read.read_train('flags')
Xt,y_test = Read.read_test('flags')
V = VK.get_V(X, y, 5)


model = VK.vknnMLTSVM(c1=1.0, c2=1, Epsi1=0.01, Epsi2=0.01, kernel='rbf', degree=2, gamma=0.3, r=0, V=V)
model.fit(X, y)
y_score = model.score(Xt)

y_pred = model.predict(Xt)

metrics = {
            "Hamming Loss": M.hamming_loss(y_test, y_pred),
            "Ranking Loss": M.ranking_loss(y_test, y_score),
             "One-Error": M.one_error(y_test, y_score),
            "Coverage": M.coverage(y_test, y_score)
    }
for metric, value in metrics.items():
    print(f"{metric}: {value:.8f}")

