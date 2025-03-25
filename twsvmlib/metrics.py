import numpy as np

def one_error(y_true, y_score):
    num_samples = y_true.shape[0]
    error_count = 0
    for i in range(num_samples):
        top_label_index = np.argmax(y_score[i])
        if y_true[i, top_label_index] != 1:
            error_count += 1
    return error_count / num_samples

def coverage(y_true, y_score):
    """
    进行了这么复杂的处理是因为birds数据集存在不少全0多标签样例
    """
    num_samples = y_true.shape[0]
    coverage_sum = 0
    wrong_samples = []
    valid_samples = 0  # 记录有效样本数（真实标签不全为0的样本）
    for i in range(num_samples):
        positive_indices = np.where(y_true[i] == 1)[0]
        # 跳过真实标签全为0的样本
        if len(positive_indices) == 0:
            # 打印位于第几行
            wrong_samples.append(i)
            continue
        # 计算覆盖误差
        ranks = np.argsort(-y_score[i])  # 得分从高到低排序的索引
        max_rank = max([np.where(ranks == idx)[0][0] for idx in positive_indices])
        coverage_sum += max_rank + 1
        valid_samples += 1
    # 避免除以0
    # print(f"wrong samples: {wrong_samples}")
    return coverage_sum / num_samples 

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