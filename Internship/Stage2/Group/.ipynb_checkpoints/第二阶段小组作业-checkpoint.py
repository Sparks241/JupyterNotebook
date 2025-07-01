import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde  # 用于计算数据密度

# 读取数据
df = pd.read_csv(r"D:\A\数据分析numpy+pandas+matplotlib\vehicles-数据.csv", encoding='gbk',low_memory=False)
'''
#第一部分
#1、
# 筛选燃油车
gas_cars = df[df['fuelType'].str.contains('|Gasoline', case=False, na=False)].copy()
gas_cars.to_csv('D:\A\数据分析numpy+pandas+matplotlib\gasoline_cars_only.csv', index=False, encoding='gbk')
# 按年份分组计算平均燃油效率
result = gas_cars.groupby('year').agg({
    'comb08': 'mean',
    'highway08': 'mean',
    'city08': 'mean'
}).reset_index()
# 重命名列名
result.columns = ['year', 'avgMPG', 'avgHghy', 'avgCity']
# 按年份排序
result = result.sort_values('year')
# 显示前几行
print(result.head())
# 保存到新 CSV
result.to_csv('D:\A\数据分析numpy+pandas+matplotlib\gas_cars_avg_fuel_efficiency.csv', index=False, encoding='gbk')

#2、绘制year和avgMPG(平均燃油率)的散点图
#读取数据
df = pd.read_csv(r"D:\A\数据分析numpy+pandas+matplotlib\gas_cars_avg_fuel_efficiency.csv", encoding='gbk')
# 查看两列的缺失值数量
missing_count = df[['year', 'avgMPG']].isnull().sum()
print("缺失值数量:\n", missing_count)
# 绘制散点图
plt.figure(1,figsize=(12, 6))  # 设置画布大小
plt.scatter(
    x=df['year'],    # x轴数据
    y=df['avgMPG'],  # y轴数据
    c='blue',          # 点颜色
    alpha=0.7,             # 透明度
    edgecolors='white',    # 边缘颜色
    s=80                  # 点大小
)
#  添加图表装饰
plt.title('年度平均燃油效率趋势（燃油车）', fontsize=15, pad=20)
plt.xlabel('年份', fontsize=12)
plt.ylabel('平均综合燃油效率 (mpg)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
# 显示图表
plt.tight_layout()
plt.show()
'''
#3、研究车的发动机排量（displ）和燃油效率（comb08）之间的关系。散点图展示。
# 查看两列的缺失值数量
missing_count = df[['displ', 'comb08']].isnull().sum()
print("缺失值数量:\n", missing_count)
#按车型分组中位数填充
df['displ'] = df.groupby('make')['displ'].transform(
    lambda x: x.fillna(x.median()) if x.notna().any() else x.fillna(df['displ'].median()))
#df['displ'] = df.groupby('make')['displ'].transform(lambda x: x.fillna(x.median()))
'''
# 过滤异常值（根据业务逻辑调整阈值）
clean_df = df[(df['displ'].between(0.5, 10)) & (df['comb08'].between(5, 150))].copy()
# 计算数据密度（优化散点图显示）
xy = np.vstack([clean_df['displ'], clean_df['comb08']])
z = gaussian_kde(xy)(xy)
#绘制散点图
plt.figure(2,figsize=(12, 7))
# 核心绘图（颜色映射数据密度）
scatter = plt.scatter(
    x=clean_df['displ'],
    y=clean_df['comb08'],
    c=z,                    # 用密度值着色
    cmap='viridis',         # 颜色映射
    alpha=0.7,              # 透明度
    s=50,                   # 点大小
    edgecolors='white',     # 边缘色
    linewidths=0.5          # 边缘线宽
)
plt.colorbar(scatter, label='数据密度')
plt.title('发动机排量与燃油效率关系', fontsize=16, pad=20)
plt.xlabel('发动机排量 (L)', fontsize=12)
plt.ylabel('综合燃油效率 (mpg)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
'''