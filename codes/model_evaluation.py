
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataPreprocessing import preprocess_data
from lightgbm import LGBMRegressor


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

        avg_rmse = np.mean(rmse_scores)
        avg_time = np.mean(times)
        results[name] = {
            "avg_rmse": avg_rmse,
            "avg_time_per_fold": avg_time
        }

        print(f"Average RMSE for {name}: {avg_rmse:.4f}")
        print(f"Average Time per Fold for {name}: {avg_time:.2f} sec")

    return results

def plot_model_comparison(results_dict, output_path="../figures/model_comparison.png"):
    # 转为 DataFrame
    df = pd.DataFrame(results_dict).T
    df = df.sort_values("avg_rmse")

    # 设置图形大小
    plt.figure(figsize=(12, 5))

    # Plot 1: RMSE
    plt.subplot(1, 2, 1)
    df['avg_rmse'].plot(kind='barh', color='skyblue')
    plt.xlabel("Average RMSE")
    plt.title("Model Comparison - RMSE")
    plt.grid(axis='x')

    # Plot 2: Time
    plt.subplot(1, 2, 2)
    df['avg_time_per_fold'].plot(kind='barh', color='orange')
    plt.xlabel("Average Time per Fold (seconds)")
    plt.title("Model Comparison - Time")
    plt.grid(axis='x')

    plt.tight_layout()

    # 保存图像为 PNG 文件
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Plot saved to: {output_path}")

def run():
    train_X, test_X, y, test_id = preprocess_data('../dataset/train.csv', '../dataset/test.csv')

    models = {
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, random_seed=42, verbose=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True,
                                        validation_fraction=0.1, n_iter_no_change=10,random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    results = evaluate_models(train_X, y, models)

    # 输出对比表格
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("avg_rmse")
    print("\nFinal Comparison:")
    print(results_df)

    plot_model_comparison(results, output_path="../figures/model_comparison.png")

if __name__ == "__main__":
    run()
