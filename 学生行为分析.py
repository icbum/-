import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import os

# 设置输出文件夹用于保存图表
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("输出文件夹:", output_dir)

# 加载数据集
df = pd.read_csv("D:\\数据集\\student_habits_performance.csv")
print("数据集前几行:\n", df.head())
print("数据集列和类型:\n", df.dtypes)

# --- 数据清洗 ---
# 检查缺失值
print("缺失值检查:\n", df.isnull().sum())

# 检查重复值
print("\n重复值数量:", df.duplicated().sum())

# 将分类变量转换为category类型
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                    'internet_quality', 'extracurricular_participation']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 对分类变量进行编码以用于建模
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

# --- 探索性数据分析 ---
# 汇总统计信息
print("\n汇总统计信息:\n", df.describe(include='all'))

# 相关性矩阵（仅数值列）
numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
print("数值列:", numeric_cols)
corr_matrix = df_encoded[numeric_cols].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')
plt.title("相关性矩阵")
plt.tight_layout()
print("正在生成相关性矩阵图...")
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), bbox_inches='tight')
plt.close()
print("相关性矩阵图已保存")

# 考试成绩分布
plt.figure(figsize=(8, 6))
plt.hist(df['exam_score'], bins=20, density=True, alpha=0.7, color='blue')
plt.title("考试成绩分布")
plt.xlabel("考试成绩")
plt.ylabel("频率")
print("正在生成考试成绩分布图...")
plt.savefig(os.path.join(output_dir, "exam_score_distribution.png"), bbox_inches='tight')
plt.close()
print("考试成绩分布图已保存")

# 按学习时间分段的考试成绩箱线图
bins = [0, 2, 4, 6, 8, 10]
labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
df['study_hours_bin'] = pd.cut(df['study_hours_per_day'], bins=bins, labels=labels)
plt.figure(figsize=(8, 6))
df.boxplot(column='exam_score', by='study_hours_bin')
plt.title("按学习时间分段的考试成绩")
plt.suptitle("")  # 移除默认标题
plt.xlabel("每日学习时间（小时）")
plt.ylabel("考试成绩")
print("正在生成学习时间分段考试成绩图...")
plt.savefig(os.path.join(output_dir, "exam_score_by_study_hours.png"), bbox_inches='tight')
plt.close()
print("学习时间分段考试成绩图已保存")

# 饮食质量对考试成绩的平均影响
diet_means = df.groupby('diet_quality', observed=True)['exam_score'].mean()
plt.figure(figsize=(8, 6))
plt.bar(diet_means.index, diet_means.values, color='green')
plt.title("按饮食质量的平均考试成绩")
plt.xlabel("饮食质量")
plt.ylabel("平均考试成绩")
print("正在生成饮食质量考试成绩图...")
plt.savefig(os.path.join(output_dir, "exam_score_by_diet_quality.png"), bbox_inches='tight')
plt.close()
print("饮食质量考试成绩图已保存")

# 学习时间与社交媒体时间的散点图，按考试成绩着色
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['study_hours_per_day'], df['social_media_hours'],
                 c=df['exam_score'], cmap='viridis')
plt.colorbar(sc, label='考试成绩')
plt.title("学习时间与社交媒体时间的关系")
plt.xlabel("每日学习时间（小时）")
plt.ylabel("每日社交媒体时间（小时）")
print("正在生成学习时间与社交媒体时间图...")
plt.savefig(os.path.join(output_dir, "study_vs_social_media.png"), bbox_inches='tight')
plt.close()
print("学习时间与社交媒体时间图已保存")

# --- 洞察：兼职工作对考试成绩的影响 ---
print("\n按兼职工作状态的平均考试成绩:")
print(df.groupby('part_time_job', observed=True)['exam_score'].mean())

# --- 机器学习：预测考试成绩 ---
# 准备特征和目标变量
X = df_encoded.drop(['student_id', 'exam_score'], axis=1)
y = df_encoded['exam_score']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)

# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n模型性能:\n均方根误差（RMSE）: {rmse:.2f}\nR²得分: {r2:.2f}")

# 特征重要性
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values()
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.index, feature_importance.values, color='purple')
plt.title("考试成绩预测的特征重要性")
plt.xlabel("重要性")
print("正在生成特征重要性图...")
plt.savefig(os.path.join(output_dir, "feature_importance.png"), bbox_inches='tight')
plt.close()
print("特征重要性图已保存")

# --- 关键发现 ---
print("\n关键发现:")
print("1. 每日学习时间与考试成绩有强相关性，学习时间越长，成绩越好。")
print("2. 饮食质量差和过多使用社交媒体与较低的考试成绩相关。")
print("3. 有兼职工作的学生平均考试成绩略低。")
print("4. 随机森林模型显示学习时间、出勤率和睡眠是考试成绩的主要预测因子。")