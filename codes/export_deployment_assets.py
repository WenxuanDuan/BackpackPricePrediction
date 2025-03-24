import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataPreprocessing import perform_feature_engineering, remove_outliers_iqr, apply_log_transform

# 保存路径
os.makedirs("../models", exist_ok=True)

# 读取数据
train = pd.read_csv("../dataset/train.csv")

# 缺失值填充
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

for col in categorical_features:
    train[col] = train[col].fillna(train[col].mode()[0])
for col in numerical_features:
    train[col] = train[col].fillna(train[col].median())

# 特征工程
train = perform_feature_engineering(train)

# 保存 weight_max
weight_max = train["Weight Capacity (kg)"].max()
joblib.dump(weight_max, "../models/weight_max.pkl")

# 异常值处理
outlier_cols = ['Weight Capacity (kg)', 'Weight_Capacity_Ratio', 'Weight_to_Compartments']
train = remove_outliers_iqr(train, outlier_cols)

# 删除无用列
train.drop(columns=["id", "Price"], inplace=True)

# 对数变换（如 apply_log_transform 中设置）
train = apply_log_transform(train, numerical_features)

# One-hot 编码
categorical_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color',
                         'Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof',
                         'Compartments_Category', 'Style_Size']
train_encoded = pd.get_dummies(train[categorical_to_encode], drop_first=False)

# 数值特征缩放
train_num = train.drop(columns=categorical_to_encode)
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_num), columns=train_num.columns)

print("Scaler columns used during training:", train_scaled)

# 保存 scaler 和数值列
joblib.dump(scaler, "../models/minmax_scaler.pkl")
joblib.dump(train_num.columns.tolist(), "../models/scaler_columns.pkl")

# 合并最终特征
train_final = pd.concat([train_encoded.reset_index(drop=True), train_scaled.reset_index(drop=True)], axis=1)

# 保存特征列
joblib.dump(train_final.columns.tolist(), "../models/reference_columns.pkl")

print("✅ All preprocessing files saved to ../models/")
