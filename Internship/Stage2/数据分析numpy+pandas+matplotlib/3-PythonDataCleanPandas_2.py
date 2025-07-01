
# coding: utf-8

# <p style="text-align:right">2017.9.1 钱小菲</p>

# # 数据合并

# ## 连接表

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                 'B': ['B0', 'B1', 'B2', 'B3'],
                 'C': ['C0', 'C1', 'C2', 'C3'],
                 'D': ['D0', 'D1', 'D2', 'D3']},
                 index=[0, 1, 2, 3])


df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                 'B': ['B4', 'B5', 'B6', 'B7'],
                 'C': ['C4', 'C5', 'C6', 'C7'],
                 'D': ['D4', 'D5', 'D6', 'D7']},
                  index=[4, 5, 6, 7])


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                 'B': ['B8', 'B9', 'B10', 'B11'],
                 'C': ['C8', 'C9', 'C10', 'C11'],
                 'D': ['D8', 'D9', 'D10', 'D11']},
                 index=[8, 9, 10, 11])


# In[3]:


df1


# In[4]:


df2


# In[5]:


df3


# In[7]:


#frames = [df1, df2, df3]
#frames
result = pd.concat([df1, df2, df3])


# In[8]:


result


# In[9]:


result = pd.concat(frames, keys=['x', 'y', 'z'])


# In[10]:


result


# In[11]:


result.index


# In[12]:


get_ipython().run_line_magic('pinfo', 'pd.concat')


# In[13]:


df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
              'D': ['D2', 'D3', 'D6', 'D7'],
              'F': ['F2', 'F3', 'F6', 'F7']},
             index=[2, 3, 6, 7])


#result = pd.concat([df1, df4], axis=1)


# In[14]:


df4


# In[15]:


df1


# In[16]:


result = pd.concat([df1, df4], axis=1)
result


# In[17]:


result = pd.concat([df1, df4], axis=1, join='inner')
result


# In[18]:


result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
result


# 使用append连接

# In[24]:


result = df1.append(df2,ignore_index=True,verify_integrity=True)
result


# In[22]:


get_ipython().run_line_magic('pinfo', 'df1.append')


# In[19]:


result = df1.append(df4,ignore_index=True)
result


# In[20]:


result = df1.append([df2, df3])
result


# 忽略连接轴上的索引

# In[27]:


result = pd.concat([df1, df4], ignore_index=True)
result


# In[21]:


result = df1.append(df4, ignore_index=True)
result


# In[23]:


s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')

result = pd.concat([df1, s1], axis=1)


# In[24]:


result


# In[46]:


s2 = pd.Series(['_0', '_1', '_2', '_3'])

result = pd.concat([df1, s2, s2, s2], axis=1)


# In[47]:


result


# In[25]:


result = pd.concat([df1, s1], axis=1, ignore_index=True)
result


# 附加行

# In[31]:


s2 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])

result = df1.append(s2, ignore_index=True)


# In[32]:


s2


# In[33]:


result


# ## 数据库风格的DataFrame连接/合并

# 单个键连接

# In[26]:


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
 

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
 


# In[27]:


left


# In[28]:


right


# In[29]:


result = pd.merge(left, right, on='key')


# In[31]:


result


# 多个键的连接

# In[33]:


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})


result = pd.merge(left, right, on=['key1', 'key2'])


# In[34]:


result


# In[35]:


result = pd.merge(left, right, how='left', on=['key1', 'key2'])
result


# In[36]:


result = pd.merge(left, right, how='right', on=['key1', 'key2'])
result


# In[37]:


result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
result


# In[38]:


left = pd.DataFrame({'A' : [1,2], 'B' : [2, 2]})

right = pd.DataFrame({'A' : [4,5,6], 'B': [2,2,2]})

result = pd.merge(left, right, on='B', how='outer')


# In[39]:


result


# 合并指示符

# In[44]:


df1 = pd.DataFrame({'col1': [0, 1], 'col_left':['a', 'b']})

df2 = pd.DataFrame({'col1': [1, 2, 2],'col_right':[2, 2, 2]})

pd.merge(df1, df2, on='col1', how='outer', indicator=True)


# 也可以使用字符串作为参数替代True，字符串将会作为该列的列名称。

# In[45]:


pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')


# 通过索引进行连接

# In[40]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])


right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


result = left.join(right)


# In[41]:


result


# In[42]:


result = left.join(right, how='outer')
result


# In[43]:


left.join(right, how='inner')


# In[29]:


pd.merge(left, right, left_index=True, right_index=True, how='outer')


# In[30]:


pd.merge(left, right, left_index=True, right_index=True, how='inner')


# 通过索引和某列连接

# In[31]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})


right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                      index=['K0', 'K1'])


result = left.join(right, on='key')


# In[32]:


result


# In[33]:


pd.merge(left, right, left_on='key', right_index=True, how='left', sort=False)


# 重叠列名称的合并

# In[34]:


left = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'v': [1, 2, 3]})

right = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'v': [4, 5, 6]})

result = pd.merge(left, right, on='k')


# In[35]:


result


# In[36]:


pd.merge(left, right, on='k', suffixes=['_l', '_r'])


# 数据框拼接

# In[48]:


df1 = pd.DataFrame([[np.nan, 3., 5.], [-4.6, np.nan, np.nan],
                    [np.nan, 7., np.nan]])
 

df2 = pd.DataFrame([[-42.6, np.nan, -8.2], [-5., 1.6, 4]],
                    index=[1, 2])


# In[44]:


df1


# In[45]:


df2


# In[38]:


df1.combine_first(df2)


# 如果都存在的值以传入的数据框为准

# In[51]:


df1.update(df2)


# In[52]:


df1


# # 数据重塑及透视表

# ## 多重索引

# In[53]:


columns = pd.MultiIndex.from_tuples([
        ('A', 'cat', 'long'), ('B', 'cat', 'long'),
        ('A', 'dog', 'short'), ('B', 'dog', 'short')],
    names=['exp', 'animal', 'hair_length'])


df = pd.DataFrame(np.random.randn(4, 4), columns=columns)


# In[44]:


df


# In[45]:


df.stack(level=['animal', 'hair_length'])


# In[46]:


df.stack(level=[1, 2])


# In[47]:


columns = pd.MultiIndex.from_tuples([('A', 'cat'), ('B', 'dog'),
                                     ('B', 'cat'), ('A', 'dog')],
                                    names=['exp', 'animal'])


index = pd.MultiIndex.from_product([('bar', 'baz', 'foo', 'qux'),
                                    ('one', 'two')],
                                   names=['first', 'second'])


df = pd.DataFrame(np.random.randn(8, 4), index=index, columns=columns)

df2 = df.iloc[[0, 1, 2, 4, 5, 7]]


# In[48]:


df2


# 可以使用level参数来调用

# In[49]:


df2.stack('exp')


# In[51]:


df2.stack('animal')


# In[52]:


df3 = df.iloc[[0, 1, 4, 7], [1, 2]]


# In[53]:


df3


# In[54]:


df3.unstack()


# In[55]:


df3.unstack(fill_value=0)


# In[58]:


df3.unstack(0)


# ## 透视表

# In[59]:


import datetime

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
                   'B': ['A', 'B', 'C'] * 8,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                   'D': np.random.randn(24),
                   'E': np.random.randn(24),
                   'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +
                        [datetime.datetime(2013, i, 15) for i in range(1, 13)]})


# In[60]:


df


# In[61]:


pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


# In[62]:


pd.pivot_table(df, values='D', index=['B'], columns=['A', 'C'], aggfunc=np.sum)


# In[63]:


pd.pivot_table(df, values=['D','E'], index=['B'], columns=['A', 'C'], aggfunc=np.sum)


# In[64]:


pd.pivot_table(df, index=['A', 'B'], columns=['C'])


# # 可视化

# Pandas内部绘图时用的是Matplotlib的API

# In[3]:


import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt


# In[4]:


print(matplotlib.style.available)


# In[67]:


matplotlib.style.use('seaborn')


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## 基本绘图：plot

# Series和DataFrame上的plot方法只是围绕plt.plot（）的简单封装。

# In[7]:


ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
#ts
ts = ts.cumsum()
#ts
ts.plot()


# 在DataFrame中，plot（）可以方便地绘制带有标签的所有列

# In[9]:


df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
#df
df = df.cumsum()

# plt.figure()
df.plot()


# 使用plot（）中的x和y关键字可以绘制一列与另一列的对应图形。

# In[12]:


df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C']).cumsum()

df3['A'] = pd.Series(list(range(len(df))))
#df3
df3.plot(x='A', y='B')


# ## 其他绘图

# 绘制方法允许除了默认线条图之外的其他绘图样式。这些方法可以通过plot（）的kind关键字参数控制。包括以下内容：  
# 
# - bar 或 barh 绘制条形图
# - hist 绘制 直方图
# - box 绘制 箱线图
# - kde 或 density 绘制 密度图
# - area 绘制 区域图
# - scatter 绘制 散点图
# - hexbin 绘制 hexagonal bin plots
# - pie 绘制 饼图

# 除了使用kind参数，你也可以使用直接对应的方法来完成相应的绘图效果。

# ### 条形图

# In[13]:


df


# In[16]:


df.iloc[5].plot(kind='bar')
plt.axhline(0, color='k')


# 调用DataFrame的plot.bar（）方法生成一个多个条形图

# In[46]:


df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
print (df2)
df2.plot.bar()


# 要生成堆叠的条形图， 传入：stacked=True:

# In[19]:


df2.plot.bar(stacked=True)


# 使用barh方法获得水平条形图：

# In[20]:


df2.plot.barh(stacked=True)


# ### 直方图

# 直方图可以通过 DataFrame.plot.hist() 和 Series.plot.hist() 进行绘制。

# In[22]:


df4 = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000),
                    'c': np.random.randn(1000) - 1}, columns=['a', 'b', 'c'])

#df4.head()
# plt.figure()

df4.plot.hist(alpha=0.5)


# In[23]:


get_ipython().run_line_magic('pinfo', 'df4.plot.hist')


# 直方图可以通过参数实现堆叠 stacked=True。 宽度大小可以通过bins关键字更改。

# In[36]:


df4.plot.hist(stacked=True, bins=20)


# ### 箱线图

# In[26]:


df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df
df.plot.box()


# 通过传递颜色关键字可以对Boxplot进行着色。

# In[27]:


color = dict(boxes='DarkGreen', whiskers='DarkOrange',
             medians='DarkBlue', caps='Gray')


df.plot.box(color=color, sym='r+')


# 通过vert来控制横向，以及通过positions控制每个图的位置

# In[28]:


df.plot.box(vert=False, positions=[1, 4, 5, 6, 8])


# 也可以通过boxplot的方法直接绘图

# In[31]:


df = pd.DataFrame(np.random.rand(10,5))
df
df.boxplot()


# 您可以使用by关键字参数创建分层的boxplot来创建分组绘制图形

# In[33]:


df = pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )

df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])
df
df.boxplot(by='X')


# In[35]:


df = pd.DataFrame(np.random.rand(10,3), columns=['Col1', 'Col2', 'Col3'])

df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])

df['Y'] = pd.Series(['A','B','A','B','A','B','A','B','A','B'])
print (df)
df.boxplot(column=['Col1','Col2'], by=['X','Y'])


# ### 面积图

# In[48]:


df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df.plot.area()


# 如果不希望绘制堆叠图，通过stacked=False，进行绘制

# In[49]:


df.plot.area(stacked=False)


# ### 散点图

# In[64]:


df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])

df.plot.scatter(x='a', y='b')


# 
# 如果希望再一个图上绘制多个列，指定绘图的目标ax后可以重复调用绘图方法，建议指定颜色和标签关键字来区分每个组。

# In[65]:


print (df)
df["e"]=["k","k","k","k","k","k","k","k","k","k"]
ax = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1')

df.plot.scatter(x='c', y='d', color='r', label='Group 2', ax=ax)


# In[57]:


get_ipython().run_line_magic('pinfo', 'df.plot.scatter')


# 关键字c可以作为列的名称给出，以为每个点提供颜色

# In[41]:


df.plot.scatter(x='a', y='b', c='c', sharex=False, s=50)


# In[42]:


df.plot.scatter(x='a', y='b', s=df['c']*200)


# ### 蜂巢图（六角形箱体图）

# In[96]:


df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
df['b'] = df['b'] + np.arange(1000)

df.plot.hexbin(x='a', y='b', sharex=False,  gridsize=25)


# ### 饼图

# In[43]:


series = pd.Series(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], name='series')

series.plot.pie(figsize=(6, 6))


# In[54]:


df = pd.DataFrame(np.random.rand(2,2), index=['a', 'b'], columns=['x', 'y'])
print (df)
df.plot.pie(subplots=True,figsize=(8, 4))


# In[44]:


df = pd.DataFrame(3 * np.random.rand(4, 2), index=['a', 'b', 'c', 'd'], columns=['x', 'y'])

df.plot.pie(subplots=True, figsize=(8, 4))


# In[45]:


series.plot.pie(labels=['AA', 'BB', 'CC', 'DD'], colors=['r', 'g', 'b', 'c'],
                autopct='%.2f', fontsize=20, figsize=(6, 6))


# 如果传递总和小于1.0的值，则matplotlib绘制半圆。

# In[100]:


series = pd.Series([0.1] * 4, index=['a', 'b', 'c', 'd'], name='series2')

series.plot.pie(figsize=(6, 6))


# ## 缺失值的默认绘制方式

# Pandas试图真实确切地描绘包含缺少值的DataFrames或Series。根据绘图类型，丢弃，省略或填充缺失值。

# <table border="1" class="docutils">
# <colgroup>
# <col width="30%">
# <col width="70%">
# </colgroup>
# <thead valign="bottom">
# <tr class="row-odd"><th class="head">Plot Type</th>
# <th class="head">NaN Handling</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr class="row-even"><td>Line</td>
# <td>Leave gaps at NaNs</td>
# </tr>
# <tr class="row-odd"><td>Line (stacked)</td>
# <td>Fill 0’s</td>
# </tr>
# <tr class="row-even"><td>Bar</td>
# <td>Fill 0’s</td>
# </tr>
# <tr class="row-odd"><td>Scatter</td>
# <td>Drop NaNs</td>
# </tr>
# <tr class="row-even"><td>Histogram</td>
# <td>Drop NaNs (column-wise)</td>
# </tr>
# <tr class="row-odd"><td>Box</td>
# <td>Drop NaNs (column-wise)</td>
# </tr>
# <tr class="row-even"><td>Area</td>
# <td>Fill 0’s</td>
# </tr>
# <tr class="row-odd"><td>KDE</td>
# <td>Drop NaNs (column-wise)</td>
# </tr>
# <tr class="row-even"><td>Hexbin</td>
# <td>Drop NaNs</td>
# </tr>
# <tr class="row-odd"><td>Pie</td>
# <td>Fill 0’s</td>
# </tr>
# </tbody>
# </table>

# 如果默认缺失值的处理方式不符合你的预期，优先手动处理缺失值（使用fillna或者dropna）。

# ## 绘图工具

# ### 矩阵散点图

# In[102]:


from pandas.plotting import scatter_matrix


# In[103]:


df = pd.DataFrame(np.random.randn(1000, 4), columns=['a', 'b', 'c', 'd'])

scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')


# ### 密度图

# In[104]:


ser = pd.Series(np.random.randn(1000))

ser.plot.kde()


# ## 绘图格式

# 大多数绘图方法都有一组关键字参数来控制返回图形的布局和格式：

# ### 控制图样

# In[107]:


df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))

df = df.cumsum()

df.plot(legend=False)


# In[108]:


df.plot()


# ### Scales

# 可以通过传入 logy 获得 log-尺度的 Y 轴。

# In[109]:


ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

ts = np.exp(ts.cumsum())

ts.plot(logy=True)


# ### 在双Y轴上面画图

# In[110]:


df.A.plot()

df.B.plot(secondary_y=True, style='g')


# 要在DataFrame中绘制一些列，请将列名称给secondary_y关键字

# In[123]:


ax = df.plot(secondary_y=['A', 'B'])

ax.set_ylabel('CD scale')

ax.right_ax.set_ylabel('AB scale')


# 注意，在辅助y轴上绘制的列在图例中自动标记为“（右）”。要关闭自动标记，请使用mark_right = False关键字：
