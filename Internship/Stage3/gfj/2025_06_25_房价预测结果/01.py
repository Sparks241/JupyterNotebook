import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
excel_file = pd.ExcelFile('house_price_regression.xlsx')
df = excel_file.parse('Sheet1')
df['title'] = df['title'].str.replace('\n', '').str.replace('\r', '').str.replace(' ', '')
# 将所有标题连接成一个文本
text = ' '.join(df['title'])
font_path_windows = 'C:/Windows/Fonts/simhei.ttf'
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path_windows).generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
# 对建筑年代进行数据清洗，提取年代数字
df['age'] = df['age'].str.extract('(\d+)').astype(float)
# 提取房屋楼层数字
df['floor_num'] = df['floor_info'].str.extract('(\d+)').astype(float)
# 定义特征和目标变量
X = df[['age', 'area', 'floor_num', 'direction', 'layout']]
y = df['price']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 对数值型和分类型特征分别进行处理
numeric_features = ['age', 'area', 'floor_num']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_features = ['direction', 'layout']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
# 创建并训练 GBDT 模型
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))])
model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = model.predict(X_test)
# 将预测结果与真实值合并为一个 DataFrame
result_df = pd.DataFrame({
    '真实房价': y_test,
    '预测房价': y_pred
})
# 绘制预测结果与真实值对比散点图
plt.figure(figsize=(12, 6))
plt.scatter(result_df['真实房价'], result_df['预测房价'], alpha=0.5)
plt.xlabel('真实房价')
plt.xticks(rotation=45)
plt.ylabel('预测房价')
plt.title('房价预测结果与真实值对比散点图')
plt.plot([result_df['真实房价'].min(), result_df['真实房价'].max()],
         [result_df['真实房价'].min(), result_df['真实房价'].max()],
         'r--', label='理想预测线')
plt.legend()
plt.show()
# 将结果保存为 Excel 文件，指定一个具体可写路径，这里以当前目录下的 output 文件夹为例，你可按需修改
import os
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
result_df.to_excel(os.path.join(output_dir, '房价预测结果.xlsx'), index=False)