import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def perform_feature_engineering(df):
    # 交叉特征
    df['Brand_Material'] = df['Brand'].astype(str) + '_' + df['Material'].astype(str)
    df['Brand_Size'] = df['Brand'].astype(str) + '_' + df['Size'].astype(str)
    df['Style_Size'] = df['Style'].astype(str) + '_' + df['Size'].astype(str)

    # 二值化处理
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})

    # 分箱（隔层数量）
    df['Compartments_Category'] = pd.cut(
        df['Compartments'],
        bins=[0, 2, 5, 10, np.inf],
        labels=['Few', 'Moderate', 'Many', 'Very Many']
    )

    # 归一化承重比 + 承重/隔层比
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)

    return df

def remove_outliers_iqr(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.15)
        Q3 = data[column].quantile(0.85)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower) & (data[column] <= upper)]
    return data

def apply_log_transform(df, numerical_features):
    skewed = df[numerical_features].skew()
    skewed_features = skewed[skewed > 0.75].index
    df[skewed_features] = np.log1p(df[skewed_features])
    return df

def preprocess_data(train_path, test_path):
    # 读取数据
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 保存 test id 和目标值
    test_id = test['id']
    y = train['Price']

    # 缺失值填充
    categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
    numerical_features = ["Weight Capacity (kg)"]

    for col in categorical_features:
        train[col] = train[col].fillna(train[col].mode()[0])
        test[col] = test[col].fillna(test[col].mode()[0])

    for col in numerical_features:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(test[col].median())

    # 特征工程
    train = perform_feature_engineering(train)
    test = perform_feature_engineering(test)

    # 删除无用列
    train.drop(['id'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    # 异常值处理（训练集）
    outlier_columns = ['Weight Capacity (kg)', 'Weight_Capacity_Ratio', 'Weight_to_Compartments']
    train = remove_outliers_iqr(train, outlier_columns)

    # 分离目标变量
    y = train['Price']
    train = train.drop(['Price'], axis=1)

    # 对数变换（处理偏态）
    train = apply_log_transform(train, numerical_features)
    test = apply_log_transform(test, numerical_features)

    # 编码处理（One-Hot）
    categorical_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color',
                             'Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof', 
                             'Compartments_Category', 'Style_Size']
    
    train_encoded = pd.get_dummies(train[categorical_to_encode], drop_first=True)
    test_encoded = pd.get_dummies(test[categorical_to_encode], drop_first=True)

    # 数值特征标准化
    train_num = train.drop(categorical_to_encode, axis=1)
    test_num = test.drop(categorical_to_encode, axis=1)

    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_num), columns=train_num.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_num), columns=test_num.columns)

    # 合并数值和编码部分
    train_final = pd.concat([train_encoded.reset_index(drop=True), train_scaled.reset_index(drop=True)], axis=1)
    test_final = pd.concat([test_encoded.reset_index(drop=True), test_scaled.reset_index(drop=True)], axis=1)

    # 对齐测试集特征（防止训练集中有额外的 one-hot 编码）
    train_final, test_final = train_final.align(test_final, join='left', axis=1, fill_value=0)

    return train_final, test_final, y, test_id
