import numpy as np
import Read
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  KFold
from tqdm.auto import tqdm
from twsvmlib import MLTSVM_k as K
from twsvmlib import metrics as M





class TqdmCallback:
    """自定义回调函数用于更新进度条"""
    def __init__(self, total_params, total_folds):
        self.pbar_params = tqdm(total=total_params, desc="参数组合", position=0)
        self.pbar_folds = tqdm(total=total_folds, desc="交叉验证", position=1)
        
    def __call__(self, res):
        self.pbar_folds.update(1)
        if res['status'] == 'finished':
            self.pbar_params.update(1)
            
    def close(self):
        self.pbar_params.close()
        self.pbar_folds.close()


if __name__ == "__main__":
    datasets = ['flags']
 
    # 外层进度条：数据集处理
    with tqdm(datasets, desc="总进度", position=0) as pbar_datasets:
        for dataset in pbar_datasets:
            pbar_datasets.set_postfix_str(f"正在处理: {dataset}")
            
            # 数据加载和预处理
            X, y = Read.read_train(dataset)
            X_test, y_test = Read.read_test(dataset)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

            # 参数范围
            c_list = np.round(np.arange(0.1, 1.1, 0.1), 2)
            gamma_list = np.round(np.arange(0.1, 2.1, 0.1), 2)
            best_score = float('inf')
            best_params = None

            # 参数搜索进度条
            with tqdm(total=len(c_list)*len(gamma_list), 
                    desc=f"{dataset} 参数搜索", 
                    position=1,
                    leave=False) as pbar_params:
                
                for c in c_list:
                    for gamma in gamma_list:
                        # 当前参数组合得分存储
                        fold_scores = []
                        
                        # 交叉验证进度条
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        with tqdm(kf.split(X), total=5, 
                                desc=f"c={c}, γ={gamma}", 
                                position=2,
                                leave=False) as pbar_folds:
                            
                            for fold, (train_idx, val_idx) in enumerate(pbar_folds):
                                X_train, X_val = X[train_idx], X[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]
                                
                                # 模型训练验证
                                model = K.kMLTSVM(c1=c, kernel='rbf', gamma=gamma)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_val)
                                
                                # 计算指标（以hamming_loss为主指标）
                                score = M.hamming_loss(y_val, y_pred)
                                fold_scores.append(score)
                                
                                # 更新折叠进度
                                pbar_folds.set_postfix_str(f"loss={score:.4f}")

                        # 计算平均得分
                        mean_score = np.mean(fold_scores)
                        
                        # 更新最佳参数
                        if mean_score < best_score:
                            best_score = mean_score
                            best_params = {'c': c, 'gamma': gamma}
                            pbar_params.set_postfix_str(f"最佳loss={best_score:.4f}")
                        
                        pbar_params.update(1)

            # 用最佳参数训练最终模型
            final_model = K.kMLTSVM(c1=best_params['c'], 
                                kernel='rbf', 
                                gamma=best_params['gamma'])
            final_model.fit(X, y)
            
            # 测试集评估
            y_pred = final_model.predict(X_test)
            y_score = final_model.score(X_test)
            
            # 计算所有指标
            metrics = {
                "Hamming Loss": M.hamming_loss(y_test, y_pred),
                "Ranking Loss": M.ranking_loss(y_test, y_score),
                "One-Error": M.one_error(y_test, y_score),
                "Coverage": M.coverage(y_test, y_score)
            }
            
            # 打印结果
            print(f"\n=== {dataset} 最终结果 ===")
            print(f"最佳参数: {best_params}")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")