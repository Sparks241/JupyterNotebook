import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
# 读取并解码文件
with open('new_data1.csv', 'rb') as f:
    data = f.read()
# 尝试多种编码解码
try:
    data = data.decode('gbk')
except UnicodeDecodeError:
    try:
        data = data.decode('latin-1')
    except UnicodeDecodeError:
        data = data.decode('utf-8', errors='ignore')
# 转换为 DataFrame
df = pd.read_csv(StringIO(data))
# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 筛选汽油类型（Regular/Premium）
gasoline_types = ['Regular', 'Premium']
gas_df = df[df['fuelType'].isin(gasoline_types)]
# 处理空数据
if gas_df.empty:
    print("警告：未找到汽油类燃料（Regular/Premium）的数据！")
    yearly_avg = pd.DataFrame(columns=['year', 'avgMPG', 'avgHghy', 'avgCity'])
else:
    # 创建副本避免链式赋值警告
    gas_df = gas_df.copy()
    # 强制转换年份为数值，过滤异常值
    gas_df.loc[:, 'year'] = pd.to_numeric(gas_df['year'], errors='coerce')
    gas_df = gas_df[gas_df['year'].between(1900, 2100)]  # 保留合理年份
    # 按年份分组（确保年份有效）
    yearly_avg = gas_df.groupby('year').agg({
        'comb08': 'mean',
        'highway08': 'mean',
        'city08': 'mean'
    }).reset_index()
    # 安全的列重命名方式
    yearly_avg = yearly_avg.rename(columns={
        'comb08': 'avgMPG',
        'highway08': 'avgHghy',
        'city08': 'avgCity'
    })
    # 按年份升序排序
    yearly_avg = yearly_avg.sort_values('year')
# 保存结果
yearly_avg.to_csv('yearly_fuel_efficiency.csv', index=False)
# 输出基本信息
rows, columns = yearly_avg.shape
print(f"\n新数据有{rows}行，{columns}列。")
if not yearly_avg.empty:
    print(f"数据包含年份范围：{yearly_avg['year'].min()}年 到 {yearly_avg['year'].max()}年")
    # 调试：打印列名和年份数据
    print("聚合后的列名：", yearly_avg.columns.tolist())
    print("当前年份数据：\n", yearly_avg[['year']])
    # 绘制燃油效率趋势图
    plt.plot(yearly_avg['year'], yearly_avg['avgMPG'], marker='o', label='综合燃油效率')
    plt.plot(yearly_avg['year'], yearly_avg['avgHghy'], marker='s', label='高速燃油效率')
    plt.plot(yearly_avg['year'], yearly_avg['avgCity'], marker='^', label='城市燃油效率')
    plt.xlabel('年份')
    plt.ylabel('平均MPG')
    plt.title('汽油车历年平均燃油效率变化趋势')
    plt.xticks(rotation=45)  # 旋转x轴标签，避免拥挤
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # 计算效率提升
    first_year = yearly_avg.iloc[0]
    last_year = yearly_avg.iloc[-1]
    mpg_increase = ((last_year['avgMPG'] - first_year['avgMPG']) / first_year['avgMPG']) * 100
    hghy_increase = ((last_year['avgHghy'] - first_year['avgHghy']) / first_year['avgHghy']) * 100
    city_increase = ((last_year['avgCity'] - first_year['avgCity']) / first_year['avgCity']) * 100
    print(f"\n综合燃油效率提升: {mpg_increase:.2f}%")
    print(f"高速燃油效率提升: {hghy_increase:.2f}%")
    print(f"城市燃油效率提升: {city_increase:.2f}%")
    # 绘制提升对比图
    efficiency_change = pd.DataFrame({
        '指标': ['综合燃油效率', '高速燃油效率', '城市燃油效率'],
        '提升百分比(%)': [mpg_increase, hghy_increase, city_increase]
    })

    bars = plt.bar(efficiency_change['指标'], efficiency_change['提升百分比(%)'],
                   color=['#4CAF50', '#2196F3', '#FF9800'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}%', ha='center', va='bottom')
    plt.title('各类型燃油效率提升百分比对比')
    plt.ylabel('提升百分比(%)')
    plt.tight_layout()
    plt.show()
else:
    print("\n数据为空，无法执行后续分析")


# 处理空数据
if gas_df.empty:
    print("警告：未找到汽油类燃料（Regular/Premium）的数据！")
else:
    # 创建副本避免链式赋值警告
    gas_df = gas_df.copy()

    # 强制转换年份为数值，过滤异常值
    gas_df.loc[:, 'year'] = pd.to_numeric(gas_df['year'], errors='coerce')
    gas_df = gas_df[gas_df['year'].between(1900, 2100)]

    # 转换displ为数值类型
    gas_df.loc[:, 'displ'] = pd.to_numeric(gas_df['displ'], errors='coerce')
    gas_df = gas_df.dropna(subset=['displ', 'comb08', 'year'])

    # 2. 绘制year和avgMPG的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(gas_df['year'], gas_df['comb08'], color='#1f77b4', alpha=0.6)
    plt.title('年份与综合燃油效率(MPG)的关系')
    plt.xlabel('年份')
    plt.ylabel('综合燃油效率(MPG)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. 研究车的马力数(displ)和燃油效率(comb08)之间的关系
    plt.figure(figsize=(10, 6))
    plt.scatter(gas_df['displ'], gas_df['comb08'], color='#ff7f0e', alpha=0.6)
    plt.title('引擎排量与综合燃油效率的关系')
    plt.xlabel('引擎排量(L)')
    plt.ylabel('综合燃油效率(MPG)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. 绘制year和马力数(displ)的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(gas_df['year'], gas_df['displ'], color='#2ca02c', alpha=0.6)
    plt.title('年份与引擎排量的关系')
    plt.xlabel('年份')
    plt.ylabel('引擎排量(L)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. 整合每年的平均燃油效率和平均马力数
    # 按年份分组计算平均燃油效率和平均引擎排量
    yearly_agg = gas_df.groupby('year').agg(
        avg_mpg=('comb08', 'mean'),
        avg_displ=('displ', 'mean')
    ).reset_index()

    # 绘制共享年份坐标轴的散点图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制平均燃油效率散点图
    color1 = '#1f77b4'
    ax1.set_xlabel('年份')
    ax1.set_ylabel('平均燃油效率(MPG)', color=color1)
    ax1.scatter(yearly_agg['year'], yearly_agg['avg_mpg'], color=color1, alpha=0.6, label='平均燃油效率')
    ax1.tick_params(axis='y', labelcolor=color1)
    # 创建第二个y轴用于平均引擎排量
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('平均引擎排量(L)', color=color2)
    ax2.scatter(yearly_agg['year'], yearly_agg['avg_displ'], color=color2, alpha=0.6, label='平均引擎排量')
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.suptitle('年份与平均燃油效率、平均引擎排量的关系')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()