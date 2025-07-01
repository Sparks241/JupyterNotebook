import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.font_manager as fm
import platform
import os

def find_suitable_font():
    """查找系统中可用的中文字体"""
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 
                    'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei',
                    'Heiti TC', 'Arial Unicode MS']
    
    for font in chinese_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return None  # 如果没有找到任何中文字体

# 设置中文字体
font = find_suitable_font()
if font:
    plt.rcParams["font.family"] = font
    print(f"已使用中文字体: {font}")
else:
    print("警告: 未找到可用的中文字体，图表中的中文可能无法正常显示")

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv("D://python实习//accepts.csv", sep=';')

# 按 bad_ind 和 bankruptcy_ind 列进行分组，并统计每组的数量
classify_df = df.groupby(['bad_ind', 'bankruptcy_ind'])[['application_id']].count().rename(columns={'application_id': '数量'})

# 创建保存图片的目录
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# 绘制柱状图展示分组统计结果
plt.figure(figsize=(10, 6))  # 设置图表大小
ax = classify_df.plot(kind='bar')

# 遍历每个柱子添加数据标签
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(height, (x + width / 2, y + height), ha='center', va='center')

# 设置标题和标签
plt.title('按不良标识和破产记录分组的申请数量')
plt.xlabel('(不良标识, 破产记录)')
plt.ylabel('申请数量')
plt.xticks(rotation=45)
plt.tight_layout()  # 调整布局，确保标签完整显示

# 保存柱状图
bar_chart_path = 'output_images/不良标识与破产记录分组统计.png'
plt.savefig(bar_chart_path)
print(f"柱状图已保存至: {bar_chart_path}")

# 显示图形
plt.show()

# 准备数据用于计算 ROC 曲线
# 将 bankruptcy_ind 进行编码
df['bankruptcy_ind_encoded'] = LabelEncoder().fit_transform(df['bankruptcy_ind'])

# 计算 FPR, TPR 和阈值
fpr, tpr, thresholds = roc_curve(df['bad_ind'], df['bankruptcy_ind_encoded'])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('不良标识与破产记录的 ROC 曲线')
plt.legend(loc='lower right')
plt.tight_layout()  # 调整布局，确保标签完整显示

# 保存ROC曲线图
roc_curve_path = 'output_images/不良标识与破产记录的ROC曲线.png'
plt.savefig(roc_curve_path)
print(f"ROC曲线图已保存至: {roc_curve_path}")

# 显示图形
plt.show()