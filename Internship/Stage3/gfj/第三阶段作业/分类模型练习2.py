import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.impute import SimpleImputer

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取CSV文件
try:
    # 使用分号分隔符读取文件
    df = pd.read_csv("accepts.csv", sep=';')
    print(f"数据基本信息：")
    df.info()
    
    # 数据预处理
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计：")
    print(missing_values[missing_values > 0])
    
    # 手动检查数据类型和内容
    print("\n数据类型和内容检查：")
    print("数据前几行：")
    print(df.head().to_string())  # 显示完整列名
    
    # 确定目标变量
    target_variable = 'bad_ind'  # 根据实际情况修改
    
    # 验证目标变量是否存在
    if target_variable not in df.columns:
        raise ValueError(f"目标变量 '{target_variable}' 不在数据列中，请检查列名。")
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 确保目标变量不在分类列列表中
    if target_variable in categorical_cols:
        categorical_cols.remove(target_variable)
    
    print(f"\n数值列 ({len(numeric_cols)}): {numeric_cols}")
    print(f"分类列 ({len(categorical_cols)}): {categorical_cols}")
    
    # 对数值型列用中位数填充
    imputer_num = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # 对分类型列用众数填充
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    
    # 编码分类变量
    for col in categorical_cols:
        print(f"编码分类列: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # 准备特征和目标变量
    X = df.drop([target_variable], axis=1)
    y = df[target_variable]
    
    # 验证特征矩阵是否为空
    if X.shape[1] == 0:
        raise ValueError("特征矩阵为空，请检查数据处理步骤。")
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 模型训练（使用随机森林分类器）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 模型预测
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.2f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 可视化
    print("开始生成可视化图表...")
    plt.figure(figsize=(15, 10))
    
    # 子图1：分类数量可视化
    plt.subplot(2, 2, 1)
    sns.countplot(x=target_variable, data=df)
    plt.title('分类数量分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    
    # 添加数值标签
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                          textcoords='offset points')
    
    # 子图2：混淆矩阵
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('实际类别')
    
    # 子图3：特征重要性
    plt.subplot(2, 2, 3)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('特征重要性')
    plt.xlabel('重要性')
    plt.tight_layout()
    
    # 子图4：ROC曲线
    plt.subplot(2, 2, 4)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    
    # 保存图像
    print("正在保存图像...")
    plt.savefig('classification_analysis.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 classification_analysis.png")
    
    # 显示图像
    print("显示可视化图表...")
    
    # 尝试设置matplotlib后端
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # 使用TkAgg后端
        plt.switch_backend('TkAgg')
    except:
        print("无法设置TkAgg后端，使用默认后端")
    
    plt.show()  # 显示图像
    
    print("程序执行完毕!")
    
except FileNotFoundError:
    print("错误：找不到'accepts.csv'文件，请确保文件在正确的路径下。")
except ValueError as ve:
    print(f"值错误: {ve}")
except Exception as e:
    print(f"发生未知错误：{e}")
    # 打印详细的堆栈信息，帮助定位问题
    import traceback
    print(traceback.format_exc())    