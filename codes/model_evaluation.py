import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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

def run():
    from dataPreprocessing import preprocess_data

    train_X, test_X, y, test_id = preprocess_data('../dataset/train.csv', '../dataset/test.csv')

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
        "CatBoost": CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
    }

    results = evaluate_models(train_X, y, models)

    # 输出对比表格
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values("avg_rmse")
    print("\nFinal Comparison:")
    print(results_df)

if __name__ == "__main__":
    run()
