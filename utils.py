import pandas as pd
import numpy as np


def preprocess_user_input(user_input, reference_columns, weight_max, minmax_scaler, scaler_columns):
    df = pd.DataFrame([user_input])

    # 🛠 特征工程
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartments_Category'] = pd.cut(
        df['Compartments'],
        bins=[0, 2, 5, 10, np.inf],
        labels=['Few', 'Moderate', 'Many', 'Very Many']
    )

    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / weight_max
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)

    # ✅ 分类特征手动指定顺序
    categories = {
        'Brand': ['Adidas', 'Jansport', 'Nike', 'Puma', 'Under Armour'],
        'Material': ['Canvas', 'Leather', 'Nylon', 'Polyester'],
        'Size': ['Large', 'Medium', 'Small'],
        'Laptop Compartment': ['Yes', 'No'],
        'Waterproof': ['Yes', 'No'],
        'Style': ['Backpack', 'Messenger', 'Tote'],
        'Color': ['Black', 'Blue', 'Gray', 'Green', 'Pink', 'Red'],
        'Compartments_Category': ['Few', 'Moderate', 'Many', 'Very Many'],
    }

    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)

    # ✅ 三个组合特征从 reference_columns 中提取类别
    combo_prefixes = ['Brand_Material', 'Brand_Size', 'Style_Size']
    for prefix in combo_prefixes:
        # 提取像 Brand_Material_Adidas_Canvas → Adidas_Canvas
        col_categories = [col.split(f"{prefix}_", 1)[1]
                          for col in reference_columns if col.startswith(f"{prefix}_")]
        df[prefix] = pd.Categorical(df[prefix], categories=sorted(set(col_categories)))

    # 🧱 One-hot 编码（所有训练时编码的列）
    onehot_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof',
                   'Style', 'Color', 'Brand_Material', 'Brand_Size',
                   'Compartments_Category', 'Style_Size']
    df_encoded = pd.get_dummies(df[onehot_cols], drop_first=False)

    # 🔢 数值特征缩放
    numeric_features = ['Compartments', 'Weight Capacity (kg)',
                        'Weight_Capacity_Ratio', 'Weight_to_Compartments']
    df_numeric = pd.DataFrame(
        minmax_scaler.transform(df[numeric_features]),
        columns=scaler_columns
    )

    # 🔗 合并所有特征
    df_combined = pd.concat([
        df_encoded.reset_index(drop=True),
        df[['Has_Laptop_Compartment', 'Is_Waterproof']].reset_index(drop=True),
        df_numeric.reset_index(drop=True)
    ], axis=1)

    # 🧩 补全缺失列并对齐顺序
    df_combined = df_combined.reindex(columns=reference_columns, fill_value=0)

    # ✅ 最后校验（可删）
    missing = list(set(reference_columns) - set(df_combined.columns))
    extra = list(set(df_combined.columns) - set(reference_columns))
    if missing:
        raise ValueError(f"❌ Still missing columns: {missing}")
    if extra:
        print(f"⚠️ Extra columns in input: {extra}")

    return df_combined
