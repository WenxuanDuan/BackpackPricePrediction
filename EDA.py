import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 创建图表保存目录
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 保存数据预览
train_data.head().to_csv(f'{output_dir}/train_head.csv', index=False)
test_data.head().to_csv(f'{output_dir}/test_head.csv', index=False)

# 数据集维度
with open(f'{output_dir}/dataset_shape.txt', 'w') as f:
    f.write(f"Training Data:\nRows: {train_data.shape[0]}, Columns: {train_data.shape[1]}\n")
    f.write(f"Test Data:\nRows: {test_data.shape[0]}, Columns: {test_data.shape[1]}\n")

# 缺失值、唯一值、数据类型
missing_values_train = pd.DataFrame({'Feature': train_data.columns,
                              '[TRAIN] No. of Missing Values': train_data.isnull().sum().values,
                              '[TRAIN] % of Missing Values': ((train_data.isnull().sum().values)/len(train_data)*100)})

missing_values_test = pd.DataFrame({'Feature': test_data.columns,
                             '[TEST] No.of Missing Values': test_data.isnull().sum().values,
                             '[TEST] % of Missing Values': ((test_data.isnull().sum().values)/len(test_data)*100)})

unique_values = pd.DataFrame({'Feature': train_data.columns,
                              'No. of Unique Values[FROM TRAIN]': train_data.nunique().values})

feature_types = pd.DataFrame({'Feature': train_data.columns,
                              'DataType': train_data.dtypes})

merged_df = pd.merge(missing_values_train, missing_values_test, on='Feature', how='left')
merged_df = pd.merge(merged_df, unique_values, on='Feature', how='left')
merged_df = pd.merge(merged_df, feature_types, on='Feature', how='left')
merged_df.to_csv(f'{output_dir}/feature_overview.csv', index=False)

# 重复值计数
with open(f'{output_dir}/duplicate_rows.txt', 'w') as f:
    f.write(f"Train duplicates: {train_data.duplicated().sum()}\n")
    f.write(f"Test duplicates: {test_data.duplicated().sum()}\n")

# 描述性统计
train_data.describe().to_csv(f'{output_dir}/train_describe.csv')

# 数值和类别变量
numerical_variables = ['Weight Capacity (kg)', 'Compartments']
target_variable = 'Price'
categorical_variables = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color']

# 自定义颜色
custom_palette = ['#3498db', '#e74c3c']
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154']
countplot_color = '#5C67A3'

# 添加 Dataset 区分训练测试集
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'

# 数值变量图像保存函数
def create_variable_plots(variable):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    sns.boxplot(
        data=pd.concat([train_data, test_data]).reset_index(drop=True),
        x=variable,
        y="Dataset",
        hue="Dataset",
        palette=custom_palette,
        ax=axes[0]
    )
    axes[0].set_title(f"Box Plot for {variable}")

    # Histogram
    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test_data, x=variable, color=custom_palette[1], kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_title(f"Histogram for {variable}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{variable}_numerical.png')
    plt.close()

for variable in numerical_variables:
    create_variable_plots(variable)

train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)

# 类别变量图像保存函数
def create_categorical_plots(variable):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3),
        startangle=140, ax=axes[0]
    )
    axes[0].set_title(f"Pie Chart for {variable}")

    sns.countplot(
        data=pd.concat([train_data, test_data]).reset_index(drop=True),
        x=variable,
        color=countplot_color,
        alpha=0.8,
        ax=axes[1]
    )
    axes[1].set_title(f"Bar Graph for {variable}")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{variable}_categorical.png')
    plt.close()

for variable in categorical_variables:
    create_categorical_plots(variable)

# 目标变量分析图
train_data['Dataset'] = 'Train'

def create_target_plots(variable):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=train_data, x=variable, y="Dataset", hue="Dataset", palette=custom_palette, ax=axes[0])
    axes[0].set_title(f"Box Plot for Target: {variable}")

    sns.histplot(data=train_data, x=variable, color=custom_palette[0], kde=True, bins=30, label="Train", ax=axes[1])
    axes[1].set_title(f"Histogram for Target: {variable}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{variable}_target.png')
    plt.close()

create_target_plots(target_variable)
train_data.drop('Dataset', axis=1, inplace=True)

# 相关性热图
train_variables = ['Compartments','Weight Capacity (kg)', 'Price']
test_variables = ['Compartments','Weight Capacity (kg)']
corr_train = train_data[train_variables].corr()
corr_test = test_data[test_variables].corr()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(corr_train, cmap='viridis', annot=True, square=True, ax=axes[0])
axes[0].set_title('Correlation Heatmap - Train')

sns.heatmap(corr_test, cmap='viridis', annot=True, square=True, ax=axes[1])
axes[1].set_title('Correlation Heatmap - Test')

plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmaps.png')
plt.close()
