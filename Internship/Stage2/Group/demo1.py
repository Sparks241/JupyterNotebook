import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置显示风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'Arial'

# 读取数据（请替换为你的文件名）
df = pd.read_csv('G:\\JupyterNotebook\\Internship\Stage2\\Group\\vehicles.csv')

# -------- 第一部分 --------

# a) 行列数
print(f"数据共有 {df.shape[0]} 行，{df.shape[1]} 列")

# b) 年份范围及每年数量
print(f"数据包含年份：{df['year'].min()} - {df['year'].max()}")
df['year'].value_counts().sort_index().plot(kind='bar')
plt.title("每年汽车数量")
plt.xlabel("年份")
plt.ylabel("数量")
plt.tight_layout()
plt.show()

# c) fuelType 类型
fuel_counts = df['fuelType'].value_counts()
print(f"燃料类型统计：\n{fuel_counts}")
fuel_counts.plot(kind='bar', title="燃料类型数量")
plt.xlabel("燃料类型")
plt.ylabel("数量")
plt.tight_layout()
plt.show()

# d) 简化 trany 字段为 Auto / Manual
def simplify_trany(x):
    if pd.isna(x):
        return 'Unknown'
    elif 'Auto' in x:
        return 'Auto'
    elif 'Manual' in x:
        return 'Manual'
    return 'Other'

df['trany_simple'] = df['trany'].apply(simplify_trany)
print(df['trany_simple'].value_counts())
sns.countplot(data=df, x='trany_simple')
plt.title("简化传动方式数量")
plt.tight_layout()
plt.show()

# e) sCharger 与年份关系（group bar）
df['sCharger'] = df['sCharger'].fillna('No')
df['sCharger_bin'] = df['sCharger'].apply(lambda x: 'Yes' if x == 'S' else 'No')
grouped = df.groupby(['year', 'sCharger_bin']).size().unstack().fillna(0)
grouped.plot(kind='bar', stacked=False)
plt.title("每年是否带增压器的汽车数量")
plt.ylabel("数量")
plt.tight_layout()
plt.show()

# f) 将 sCharger 转为布尔型
df['sCharger_bool'] = df['sCharger'] == 'S'

# -------- 第二部分 --------

# 只研究燃油车
gas_df = df[df['fuelType1'] == 'Gasoline']

# 1) 每年平均 MPG/高速/城市
avg_mpg = gas_df.groupby('year')[['comb08', 'highway08', 'city08']].mean().reset_index()
avg_mpg.columns = ['year', 'avgMPG', 'avgHghy', 'avgCity']
print(avg_mpg.head())

# 2) year vs avgMPG
sns.scatterplot(data=avg_mpg, x='year', y='avgMPG')
plt.title("年份 vs 平均MPG")
plt.tight_layout()
plt.show()

# 3) 排量与 MPG 的关系
sns.scatterplot(data=gas_df, x='displ', y='comb08')
plt.title("排量 vs 综合MPG")
plt.xlabel("排量(L)")
plt.ylabel("综合MPG")
plt.tight_layout()
plt.show()

# 4) year vs 排量
sns.scatterplot(data=gas_df, x='year', y='displ')
plt.title("年份 vs 排量")
plt.tight_layout()
plt.show()

# 5) 整合 avgMPG 和 avgDispl
avg_displ = gas_df.groupby('year')['displ'].mean().reset_index(name='avgDispl')
merged = pd.merge(avg_mpg, avg_displ, on='year')

fig, ax1 = plt.subplots()
sns.lineplot(x='year', y='avgMPG', data=merged, ax=ax1, label='平均MPG', color='blue')
ax1.set_ylabel("平均MPG", color='blue')
ax2 = ax1.twinx()
sns.lineplot(x='year', y='avgDispl', data=merged, ax=ax2, label='平均排量', color='green')
ax2.set_ylabel("平均排量(L)", color='green')
plt.title("平均燃油效率与排量随年份变化")
plt.tight_layout()
plt.show()

# -------- 第三部分 --------

# 1) 哪种气缸数最多
cyl_counts = df['cylinders'].value_counts()
print(f"气缸数分布：\n{cyl_counts}")
cyl_counts.plot(kind='bar', title="不同气缸数的车辆数量")
plt.xlabel("气缸数")
plt.ylabel("数量")
plt.tight_layout()
plt.show()

# 2) 4缸车的制造商数量随年份变化
df_4cyl = df[df['cylinders'] == 4]
make_year = df_4cyl.groupby(['year', 'make']).size().reset_index(name='count')

# 只展示每年制造商数量前10的品牌
top_makes = df_4cyl['make'].value_counts().nlargest(10).index
make_year_top = make_year[make_year['make'].isin(top_makes)]

plt.figure(figsize=(14,6))
sns.lineplot(data=make_year_top, x='year', y='count', hue='make')
plt.title("4缸汽车每年主要制造商数量")
plt.ylabel("数量")
plt.tight_layout()
plt.show()
