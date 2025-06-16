import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体（Windows 自带）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

# 1. 加载数据
df = pd.read_csv("D:\\数据集\\winequality-red.csv")

# 2. 数据清洗与检查
# 检查是否有缺失值
print("缺失值统计:\n", df.isnull().sum())
# 检查数据类型
print("\n数据类型:\n", df.dtypes)
# 显示基本统计信息
print("\n数据描述:\n", df.describe())

# 3. 探索性数据分析 (EDA)
# 绘制质量评分的分布直方图
plt.figure(figsize=(8, 5))
plt.hist(df['quality'], bins=range(3, 9), edgecolor='black', alpha=0.7)
plt.xlabel('质量评分')
plt.ylabel('样本数量')
plt.title('红酒质量评分分布')
plt.grid(True, alpha=0.3)
plt.show()

# 绘制酒精含量与质量的散点图
plt.figure(figsize=(8, 5))
plt.scatter(df['alcohol'], df['quality'], alpha=0.5, color='purple')
plt.xlabel('酒精含量 (% 体视)')
plt.ylabel('质量评分')
plt.title('酒精含量与红酒质量的关系')
plt.grid(True, alpha=0.3)
plt.show()

# 计算并显示特征与质量的相关性
correlation = df.corr()['quality'].sort_values(ascending=False)
print("\n与质量评分的相关性:\n", correlation)

# 4. 机器学习建模
# 准备特征和目标变量
X = df.drop('quality', axis=1).values  # 所有化学属性
y = df['quality'].values  # 质量评分

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化（使用 NumPy）
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n随机森林模型均方根误差 (RMSE):", rmse)

# 特征重要性
feature_names = df.drop('quality', axis=1).columns
feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\n特征重要性:\n", feature_importance)

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='teal')
plt.title('随机森林模型中的特征重要性')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.tight_layout()
plt.show()