import pandas as pd
import numpy as np
from codes.dataPreprocessing import perform_feature_engineering

def preprocess_user_input(input_dict, reference_columns, weight_max, scaler):
    # 将 dict 转为 DataFrame
    df = pd.DataFrame([input_dict])

    # 类型转换
    df['Compartments'] = df['Compartments'].astype(float)
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].astype(float)

    # 特征工程（保持与训练一致）
    df = perform_feature_engineering(df)

    # One-Hot 编码
    categorical_features = [
        'Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color',
        'Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment', 'Is_Waterproof',
        'Compartments_Category', 'Style_Size'
    ]
    df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

    # 数值特征
    df_num = df.drop(columns=categorical_features)
    df_num['Weight Capacity (kg)'] = np.log1p(df_num['Weight Capacity (kg)'])  # 和训练保持一致

    # 使用训练时保存的 MinMaxScaler 对数值特征进行缩放
    df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns)

    # 合并
    df_final = pd.concat([df_encoded.reset_index(drop=True), df_num_scaled.reset_index(drop=True)], axis=1)

    # 对齐列（补全缺失列为 0）
    df_final = df_final.reindex(columns=reference_columns, fill_value=0)

    return df_final
