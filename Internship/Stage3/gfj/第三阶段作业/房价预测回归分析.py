import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def find_suitable_font():
    """查找系统中可用的中文字体"""
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 
                    'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei',
                    'Heiti TC', 'Arial Unicode MS']
    
    for font in chinese_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    return None  # 如果没有找到任何中文字体

def regression_analysis_and_visualization(file_path):
    # 尝试查找系统中可用的中文字体
    font = find_suitable_font()
    
    # 设置matplotlib
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
    
    if font:
        print(f"找到可用中文字体: {font}")
        plt.rcParams['font.family'] = font
    else:
        print("未找到中文字体，将使用系统默认字体。如果显示异常，请手动安装中文字体。")
    
    # 加载数据集
    df = pd.read_csv(file_path)

    # 提取特征和目标变量
    X = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers']]
    y = df['Price']

    # 构建线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 模型预测
    y_pred = model.predict(X)

    # 模型评估
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # 输出回归系数和截距
    print('回归系数：', model.coef_)
    print('截距：', model.intercept_)
    print('R 平方值：', r2)
    print('均方误差：', mse)
    
    # 计算残差（修复：将残差计算移到使用之前）
    residuals = y - y_pred

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制实际值与预测值的散点图
    scatter1 = ax1.scatter(y, y_pred, alpha=0.7, color='dodgerblue')
    ax1.set_xlabel('实际价格', fontsize=12)
    ax1.set_ylabel('预测价格', fontsize=12)
    ax1.set_title('实际价格与预测价格的散点图', fontsize=14, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 添加拟合线
    fit_line = ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.legend([scatter1, fit_line[0]], ['数据点', '理想拟合线'], loc='upper left')

    # 绘制残差图
    scatter2 = ax2.scatter(y_pred, residuals, alpha=0.7, color='salmon')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('预测价格', fontsize=12)
    ax2.set_ylabel('残差', fontsize=12)
    ax2.set_title('残差图', fontsize=14, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 添加统计信息文本
    stats_text = f'R² = {r2:.4f}\nMSE = {mse:.2e}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 调整布局
    plt.tight_layout(pad=3.0)  # 增加子图之间的间距
    plt.subplots_adjust(top=0.85)  # 调整顶部间距

    plt.show()

# 请将下面的文件路径替换为你的文件路径
file_path = "D:\python实习\house-prices.csv"
regression_analysis_and_visualization(file_path)