from statistics import correlation

import  pandas as pd
import  numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:\\ins收藏的\\Iris.csv")
print(df.head(8))#显示前8行信息，默认前五行
print(df.info())
print(df.isnull().sum())
print(df.describe())

features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
print('按花种统计指标：\n')
print(df.groupby('Species')[features].agg(['mean','std']))

np_data=df[features].to_numpy()
correlation_data=np.corrcoef(np_data,rowvar=False)
print('\n特征相关性矩阵分析：')
print(correlation_data)


#画图，未完成！！
# 3. 可视化
# 3.1 萼片长度分布直方图（按花种）
plt.figure(figsize=(10, 6))
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    plt.hist(subset['SepalLengthCm'], bins=10, alpha=0.5, label=species, edgecolor='black')
plt.title('Sepal Length Distribution by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3.2 花瓣长度与花瓣宽度的散点图（按花种着色）
plt.figure(figsize=(10, 6))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'],
                c=colors[species], label=species, s=50)
plt.title('Petal Length vs Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3.3 每个特征均值的柱状图比较
mean_values = df.groupby('Species')[features].mean()
species = df['Species'].unique()
x = np.arange(len(species))  # 柱状图的 x 轴位置
width = 0.2  # 柱宽

plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.bar(x + i * width, mean_values[feature], width, label=feature)
plt.title('Mean Feature Values by Species')
plt.xlabel('Species')
plt.ylabel('Mean Value (cm)')
plt.xticks(x + width * 1.5, species)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

