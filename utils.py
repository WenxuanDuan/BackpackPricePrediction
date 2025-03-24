import pandas as pd
import numpy as np
from codes.dataPreprocessing import perform_feature_engineering

def preprocess_user_input(input_dict, reference_columns, weight_max):
    # 将 dict 转为 DataFrame
    df = pd.DataFrame([input_dict])

    # 类型转换
    df['Compartments'] = df['Compartments'].astype(float)
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].astype(float)

    # 特征工程（跟训练时一致）
    df = perform_feature_engineering(df)

    # One-Hot 编码（注意 drop_first=True）
    categorical_features = [
        'Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color',
        'Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment', 'Is_Waterproof',
        'Compartments_Category', 'Style_Size'
    ]
    df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

    # 数值特征（无需 log，仅归一化）
    df_num = df.drop(categorical_features, axis=1)
    df_num['Weight Capacity (kg)'] = np.log1p(df_num['Weight Capacity (kg)'])  # 跟训练保持一致
    df_num['Weight_Capacity_Ratio'] = df_num['Weight Capacity (kg)'] / np.log1p(weight_max)
    df_num['Weight_to_Compartments'] = df_num['Weight Capacity (kg)'] / (df_num['Compartments'] + 1)

    # 合并
    df_final = pd.concat([df_encoded, df_num.reset_index(drop=True)], axis=1)

    # 对齐训练集的列（缺失的填 0）
    df_final = df_final.reindex(columns=reference_columns, fill_value=0)

    return df_final