import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from dataPreprocessing import preprocess_data

# 路径
train_path = "../dataset/train.csv"
test_path = "../dataset/test.csv"

# 预处理
train_X, test_X, y, test_id = preprocess_data(train_path, test_path)

# 保存 reference_columns
reference_columns = train_X.columns.tolist()
joblib.dump(reference_columns, "../models/reference_columns.pkl")

# 保存 weight_max
weight_max = pd.read_csv(train_path)["Weight Capacity (kg)"].max()
joblib.dump(weight_max, "../models/weight_max.pkl")

# 保存 scaler（用在数值特征）
numerical_columns = train_X.select_dtypes(include='number').columns.tolist()
scaler = MinMaxScaler()
scaler.fit(train_X[numerical_columns])
joblib.dump(scaler, "../models/minmax_scaler.pkl")
joblib.dump(numerical_columns, "../models/scaler_columns.pkl")  # ✅ 保存列名顺序


print("✅ Saved: reference_columns.pkl, weight_max.pkl, minmax_scaler.pkl")
