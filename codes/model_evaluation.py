import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from dataPreprocessing import preprocess_data


# 模型评估函数
def evaluate_models(train_X, y, models, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        rmse_scores = []
        times = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_X), 1):
            X_train, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            start = time.time()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            end = time.time()

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            elapsed = end - start
            rmse_scores.append(rmse)
            times.append(elapsed)

            print(f"  Fold {fold} - RMSE: {rmse:.4f} | Time: {elapsed:.2f} sec")

        results[name] = {
            "avg_rmse": np.mean(rmse_scores),
            "avg_time_per_fold": np.mean(times)
        }

        print(f"Average RMSE for {name}: {np.mean(rmse_scores):.4f}")
        print(f"Average Time per Fold for {name}: {np.mean(times):.2f} sec")

    return results


# 模型对比图保存函数
def plot_model_comparison(results_dict, output_path="../figures/model_comparison.png"):
    df = pd.DataFrame(results_dict).T.sort_values("avg_rmse")

    plt.figure(figsize=(12, 5))

    # RMSE plot
    plt.subplot(1, 2, 1)
    df['avg_rmse'].plot(kind='barh', color='skyblue')
    plt.xlabel("Average RMSE")
    plt.title("Model Comparison - RMSE")
    plt.grid(axis='x')

    # Time plot
    plt.subplot(1, 2, 2)
    df['avg_time_per_fold'].plot(kind='barh', color='orange')
    plt.xlabel("Average Time per Fold (seconds)")
    plt.title("Model Comparison - Time")
    plt.grid(axis='x')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to: {output_path}")


# 主程序
def run():
    train_X, test_X, y, test_id = preprocess_data('../dataset/train.csv', '../dataset/test.csv')

    models = {
        # "KNN (k=5)": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        # "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, random_seed=42, verbose=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=10, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    # 基础模型评估
    results = evaluate_models(train_X, y, models)

    # 转为 DataFrame 以便排序和获取 Top 3 模型名
    results_df = pd.DataFrame(results).T.sort_values("avg_rmse")
    top3_names = results_df.head(3).index.tolist()
    print(f"\nTop 3 models for voting: {top3_names}")

    # 构造 VotingRegressor
    top3_models = [(name, models[name]) for name in top3_names]
    voting_model = VotingRegressor(estimators=top3_models)

    # 对 VotingRegressor 做评估
    print("\nEvaluating: VotingRegressor (Top 3)")
    rmse_scores = []
    times = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_X), 1):
        X_train, X_val = train_X.iloc[train_idx], train_X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        start = time.time()
        voting_model.fit(X_train, y_train)
        preds = voting_model.predict(X_val)
        end = time.time()

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        elapsed = end - start
        rmse_scores.append(rmse)
        times.append(elapsed)
        print(f"  Fold {fold} - RMSE: {rmse:.4f} | Time: {elapsed:.2f} sec")

    results["VotingRegressor (Top 3)"] = {
        "avg_rmse": np.mean(rmse_scores),
        "avg_time_per_fold": np.mean(times)
    }

    # 输出最终比较表格 & 图
    final_results_df = pd.DataFrame(results).T.sort_values("avg_rmse")
    print("\nFinal Comparison:")
    print(final_results_df)

    plot_model_comparison(results, output_path="../figures/model_comparison.png")


if __name__ == "__main__":
    run()



