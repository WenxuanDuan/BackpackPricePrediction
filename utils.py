# ✅ utils.py

import pandas as pd
import numpy as np
import joblib
from codes.dataPreprocessing import perform_feature_engineering


def preprocess_user_input(input_dict, reference_columns, weight_max, scaler, scaler_columns):
    df = pd.DataFrame([input_dict])

    # 类型转换
    df['Compartments'] = df['Compartments'].astype(float)
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].astype(float)

    # 特征工程
    df = perform_feature_engineering(df)

    # 类别特征（One-Hot）
    categorical_features = [
        'Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color',
        'Brand_Material', 'Brand_Size',
        'Compartments_Category', 'Style_Size'
    ]
    binary_features = ['Has_Laptop_Compartment', 'Is_Waterproof']

    # One-hot 编码
    df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

    # 数值特征处理（包括 binary 特征）
    df_num = df.drop(categorical_features, axis=1)
    df_num['Weight Capacity (kg)'] = np.log1p(df_num['Weight Capacity (kg)'])
    df_num['Weight_Capacity_Ratio'] = df_num['Weight Capacity (kg)'] / np.log1p(weight_max)
    df_num['Weight_to_Compartments'] = df_num['Weight Capacity (kg)'] / (df_num['Compartments'] + 1)

    # 按 scaler_columns 顺序排列
    df_num = df_num[scaler_columns]

    # 数值归一化
    num_scaled = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns)

    # 合并
    df_final = pd.concat([df_encoded, num_scaled.reset_index(drop=True)], axis=1)
    df_final = df_final.reindex(columns=reference_columns, fill_value=0)

    return df_final

