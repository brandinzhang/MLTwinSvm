import numpy as np
from sklearn.preprocessing import StandardScaler
from MLTSVM_ova import ovaMLTSVM
import Read

def one_error(y_true, y_score):
    num_samples = y_true.shape[0]
    error_count = 0
    for i in range(num_samples):
        top_label_index = np.argmax(y_score[i])
        if y_true[i, top_label_index] != 1:
            error_count += 1
    return error_count / num_samples

def coverage(y_true, y_score):
    num_samples = y_true.shape[0]
    coverage_sum = 0
    valid_samples = 0  # 记录有效样本数（真实标签不全为0的样本）
    for i in range(num_samples):
        positive_indices = np.where(y_true[i] == 1)[0]
        # 跳过真实标签全为0的样本
        if len(positive_indices) == 0:
            # 打印位于第几行
            print(i)
            continue
        # 计算覆盖误差
        ranks = np.argsort(-y_score[i])  # 得分从高到低排序的索引
        max_rank = max([np.where(ranks == idx)[0][0] for idx in positive_indices])
        coverage_sum += max_rank + 1
        valid_samples += 1
    # 避免除以0
    return coverage_sum / valid_samples if valid_samples > 0 else 0.0

def ranking_loss(y_true, y_score):
    num_samples = y_true.shape[0]
    loss = 0
    for i in range(num_samples):
        pos = np.where(y_true[i] == 1)[0]
        neg = np.where(y_true[i] == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cnt = 0
        for p in pos:
            for n in neg:
                if y_score[i][n] >= y_score[i][p]:
                    cnt += 1
        loss += cnt / (len(pos) * len(neg))
    return loss / num_samples

# hamm_loss 函数
def hamming_loss(y_true, y_pred):
    num_samples = y_true.shape[0]
    loss = 0
    for i in range(num_samples):
        # 计算汉明损失
        loss += np.sum(y_true[i] != y_pred[i])
    return loss / (num_samples * y_true.shape[1])

if __name__ == "__main__":
    # 定义不同的参数组合
    params_list = [
        {'c1': 1, 'c2': 1, 'Epsi1': 0.01, 'Epsi2': 0.01, 'kernel': 'linear', 'degree': 2, 'gamma': 1.0, 'r': 0},
        {'c1': 2, 'c2': 2, 'Epsi1': 0.001, 'Epsi2': 0.001, 'kernel': 'rbf', 'degree': 2, 'gamma': 0.1, 'r': 0},
        {'c1': 0.5, 'c2': 0.5, 'Epsi1': 0.1, 'Epsi2': 0.1, 'kernel': 'poly', 'degree': 3, 'gamma': 0.5, 'r': 1}
    ]

    # 读取数据集
    X, y = Read.read_train('flags')
    Xt, yt = Read.read_test('flags')
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xt = scaler.transform(Xt)

    best_params = None
    best_hamming_loss = float('inf')
    results = []

    # 遍历所有参数组合
    for params in params_list:
        model = ovaMLTSVM(**params)
        model.fit(X, y)

        y_pred = model.predict(Xt)
        y_score = model.score(Xt)

        oe = one_error(yt, y_score)
        cov = coverage(yt, y_score)
        rl = ranking_loss(yt, y_score)
        hl = hamming_loss(yt, y_pred)

        results.append({
            'params': params,
            'one_error': oe,
            'coverage': cov,
            'ranking_loss': rl,
            'hamming_loss': hl
        })

        # 根据汉明损失更新最优参数
        if hl < best_hamming_loss:
            best_hamming_loss = hl
            best_params = params

    # 输出所有结果
    for result in results:
        print(f"Params: {result['params']}")
        print(f"One Error: {result['one_error']}")
        print(f"Coverage: {result['coverage']}")
        print(f"Ranking Loss: {result['ranking_loss']}")
        print(f"Hamming Loss: {result['hamming_loss']}")
        print("-" * 50)

