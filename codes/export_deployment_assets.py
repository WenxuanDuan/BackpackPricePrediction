import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from dataPreprocessing import perform_feature_engineering, apply_log_transform

# 确保保存路径存在
os.makedirs("models", exist_ok=True)

# === 1. 读取训练数据 ===
df = pd.read_csv("../dataset/train.csv")

# 仅保留有用列，避免太大
columns_to_use = ['Brand', 'Material', 'Size', 'Style', 'Color',
                  'Laptop Compartment', 'Waterproof', 'Compartments', 'Weight Capacity (kg)', 'Price', 'id']
df = df[columns_to_use].copy()

# === 2. 填补缺失值 ===
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in numerical_features:
    df[col] = df[col].fillna(df[col].median())

# === 3. 特征工程 ===
df = perform_feature_engineering(df)

# === 4. 提取归一化最大值 ===
weight_max = df["Weight Capacity (kg)"].max()
joblib.dump(weight_max, "../models/weight_max.pkl")

# === 5. 提取 target，删除无关列 ===
df.drop(["Price", "id"], axis=1, inplace=True)

# === 6. Apply log transform (optional) ===
df = apply_log_transform(df, numerical_features)

# === 7. One-hot 编码 ===
categorical_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color',
                         'Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof',
                         'Compartments_Category', 'Style_Size']
df_encoded = pd.get_dummies(df[categorical_to_encode], drop_first=True)

# 数值列标准化
df_num = df.drop(categorical_to_encode, axis=1)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

# 合并编码 + 数值
df_final = pd.concat([df_encoded.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

# === 8. 保存 reference columns & scaler ===
reference_columns = df_final.columns.tolist()
joblib.dump(reference_columns, "../models/reference_columns.pkl")
joblib.dump(scaler, "../models/minmax_scaler.pkl")

print("✅ Exported for deployment: stacking_model.pkl + reference_columns.pkl + weight_max.pkl + minmax_scaler.pkl")
