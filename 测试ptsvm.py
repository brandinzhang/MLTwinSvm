import numpy as np
import Read
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from twsvmlib import MLTSVM_p as P
from twsvmlib import metrics as M

if __name__ == "__main__":
    datasets = ['emotions', 'birds', 'yeast', 'flags']
    
    # 外层进度条：数据集处理
    with tqdm(datasets, desc="总进度") as pbar_datasets:
        for dataset in pbar_datasets:
            pbar_datasets.set_postfix_str(f"正在处理: {dataset}")
            
            # 数据加载和预处理
            X, y = Read.read_train(dataset)
            X_test, y_test = Read.read_test(dataset)
            
            # 划分训练集和验证集（70%训练，30%验证）
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=215
            )
            
            # 标准化处理（仅在训练集上fit）
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # 参数范围（0.1到1.0，步长0.1）
            c_list = np.round(np.arange(0.1, 1.1, 0.1), 2)
            gamma_list = np.round(np.arange(0.1, 1.1, 0.1), 2)
            best_score = float('inf')
            best_params = None

            # 参数搜索进度条（双重循环）
            with tqdm(total=len(c_list)*len(gamma_list), 
                     desc=f"{dataset} 参数搜索",
                     leave=False) as pbar_params:
                for c in c_list:
                    for gamma in gamma_list:
                        # 训练模型
                        model = P.pMLTSVM(c1=c, kernel='rbf', gamma=gamma)
                        model.fit(X_train, y_train)
                        
                        # 验证集预测
                        y_pred = model.predict(X_val)
                        y_score = model.score(X_val)
                        
                        # 计算四个指标之和
                        hamming = M.hamming_loss(y_val, y_pred)
                        ranking = M.ranking_loss(y_val, y_score)
                        one_error = M.one_error(y_val, y_score)
                        coverage = M.coverage(y_val, y_score)
                        total_score = hamming + ranking + one_error + coverage
                        
                        # 更新最佳参数
                        if total_score < best_score:
                            best_score = total_score
                            best_params = {'c': c, 'gamma': gamma}
                            pbar_params.set_postfix_str(f"当前最佳: {best_score:.4f}")
                        
                        pbar_params.update(1)

            # 合并训练集和验证集重新训练
            X_full = np.vstack((X_train, X_val))
            y_full = np.vstack((y_train, y_val))
            
            # 最终模型训练
            final_model = P.pMLTSVM(
                c1=best_params['c'], 
                kernel='rbf', 
                gamma=best_params['gamma']
            )
            final_model.fit(X_full, y_full)
            
            # 测试集评估
            y_pred = final_model.predict(X_test)
            y_score = final_model.score(X_test)
            
            # 计算所有指标
            metrics = {
                "Hamming Loss": M.hamming_loss(y_test, y_pred),
                "Ranking Loss": M.ranking_loss(y_test, y_score),
                "One-Error": M.one_error(y_test, y_score),
                "Coverage": M.coverage(y_test, y_score),
                "Total Score": (
                    M.hamming_loss(y_test, y_pred) + 
                    M.ranking_loss(y_test, y_score) + 
                    M.one_error(y_test, y_score) + 
                    M.coverage(y_test, y_score)
                )
            }
            
            # 打印结果
            print(f"\n=== {dataset} 结果 ===")
            print(f"最佳参数: {best_params}")
            for name, value in metrics.items():
                print(f"{name}: {value:.4f}")