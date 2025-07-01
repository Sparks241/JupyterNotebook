
# coding: utf-8

# # Pandas对象

# Series：一维数组，与Numpy中的一维array类似。二者与Python基本的数据结构List也很相近，其区别是：List中的元素可以是不同的数据类型，而Array和Series中则只允许存储相同的数据类型，这样可以更有效的使用内存，提高运算效率。
# Time- Series：以时间为索引的Series。
# DataFrame：二维的表格型数据结构。很多功能与R中的data.frame类似。可以将DataFrame理解为Series的容器。以下的内容主要以DataFrame为主。
# Panel ：三维的数组，可以理解为DataFrame的容器
# 
# 
# Pandas 本身的内容比较多，和NumPy的关联也非常紧密。在这里我们主要讲常用的两个数据结构(DataFrame和Series)和相应的常用方法属性等。

# In[1]:


import numpy as np
import pandas as pd


# 在pandas的数据结构中，数据对齐是内在的。标签（即索引）和数据之间的链接不会被破坏，除非你显式地进行相关的操作。

# ## Series

# Series数据结构是一种类似于一维数组的对象，是由一组数据（各种Numpy数据类型）以及一组与之相关的标签（即索引）组成。

# ### 创建Series

# 多数情况下，Series数据结构是我们直接从DataFrame数据结构中截取出来的，但也可以自己创建Series。语法如下：
# ```
# s = pd.Series(data, index=index)
# ```
# 
# 其中data可以是不同的内容：
# - 字典
# - ndarray
# - 标量
# 
# index 是轴标签列表，根据不同的情况传入的内容有所不同。

# ##### 由ndarray构建

# 如果data是ndarray，则索引的长度必须与数据相同。如果没有入索引，将创建一个值为[0，...，len（data）-1]的索引。

# In[7]:


ser = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
ser
#type(ser)


# In[3]:


ser.index


# In[4]:


ser.index[[True,False,True,True,True]]


# In[5]:


pd.Series(np.random.randn(5))


# In[10]:


np.random.seed(100)
ser=pd.Series(np.random.rand(7)); ser


# In[8]:


import calendar as cal
monthNames=[cal.month_name[i] for i in np.arange(1,6)]
print (monthNames)
months=pd.Series(np.arange(1,6),index=monthNames);
months


# In[8]:


months.index


# ##### 由字典构建

# 若data是一个dict，如果传递了索引，则索引中与标签对应的数据中的值将被列出。否则，将从dict的排序键构造索引（如果可能）。

# In[9]:


d = {'a' : 0., 'b' : 1., 'c' : 2.}
pd.Series(d)


# In[10]:


pd.Series(d, index=['b', 'c', 'd', 'a'])


# In[11]:


stockPrices = {'GOOG':1180.97,'FB':62.57,'TWTR': 64.50, 'AMZN':358.69,'AAPL':500.6}
stockPriceSeries=pd.Series(stockPrices,index=['GOOG','FB','YHOO','TWTR','AMZN','AAPL'],name='stockPrices')
stockPriceSeries


# 注：NaN(not a number)是Pandas的标准缺失数据标记。

# In[14]:


print(stockPriceSeries.name)
stockPriceSeries.index


# In[15]:


dogSeries=pd.Series('chihuahua',index=['breed','countryOfOrigin','name', 'gender'])
dogSeries


# ##### 由标量创建

# 如果数据是标量值，则必须提供索引。将该值重复以匹配索引的长度。

# In[16]:


pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])


# 除了上述之外，类ndarray的对象传入后也会转换为ndarray来创建Series

# In[17]:


ser = pd.Series([5,4,2,-3,True])
ser


# In[3]:


ser11 = pd.Series([True,True])
ser11


# In[18]:


ser.values


# In[19]:


ser.index


# In[25]:


ser2 = pd.Series([5, 4, 2, -3, True], index=['b', 'e', 'c', 'a', 'd'])
ser2


# In[26]:


ser2.index


# In[27]:


ser2.values


# ### Series is ndarray-like

# Series与ndarray非常相似，是大多数NumPy函数的有效参数。包括像切片这样的索引操作。

# In[20]:


ser = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
ser


# In[21]:


ser[0]


# In[22]:


ser[:3]


# In[23]:


ser[ser > 0]


# In[24]:


ser[ser > ser.median()]


# In[25]:


ser[ser > ser.median()]=1
ser


# In[26]:


ser[[4, 3, 1]]


# In[27]:


np.exp(ser)


# ### Series is dict-like

# Series同时也像一个固定大小的dict，可以通过索引标签获取和设置值：

# In[28]:


ser


# In[29]:


ser['a']


# In[30]:


ser['e'] = 12.
ser


# In[31]:


'e' in ser


# In[32]:


'f' in ser


# 注：如果引用了未包含的标签，则会引发异常：

# 使用get方法，未包含的索引则会返回None，或者特定值。和dict的操作类似。

# In[34]:


print(ser.get('f'))
#print(ser.get('a'))


# In[47]:


ser.get('f', np.nan)


# ### 矢量化操作&标签对齐

# 在进行数据分析时，通常没必要去使用循环，而是使用矢量化的操作方式。

# In[35]:


ser


# In[36]:


ser + ser


# In[37]:


ser * 2


# In[38]:


np.exp(ser)


# Series和ndarray之间的一个主要区别是，Series之间的操作会自动对齐基于标签的数据。

# In[39]:


ser


# In[40]:


ser[1:] + ser[:-1]


# 未对齐Series之间的操作结果将包含所涉及的索引的并集。如果在其中一个Seires中找不到标签，结果将被标记为NaN。

# 注意：通常不同索引对象之间的操作的默认结果产生索引的并集，以避免信息丢失。  
# 因为尽管数据丢失，但拥有索引标签也可以作为计算的重要信息。当然也可以选择通过dropna功能删除丢失数据的标签。

# ### 属性

# 名称属性：

# In[41]:


s = pd.Series(np.random.randn(5), name='something')


# In[54]:


s


# In[42]:


s.name


# 在多数情况下，series名称会被自动分配，例如在获取1D切片的DataFrame时。（后续DataFrame操作将会讲解到）

# In[43]:


s2 = s.rename("different")
s2


# 这里需要注意的是，s和s2是指向不同的对象的。

# 通过索引属性获取索引

# In[44]:


s.index


# 索引对象也有一个name属性

# In[45]:


s.index.name = "index_name"
s


# 通过值索引获取值

# In[46]:


s.values


# ## DataFrame

# DataFrame 是可以包含不同类型的列且带索引的二维数据结构，类似于SQL表，或者Series的字典集合。

# ### 创建DataFrame

# DataFrame 是被使用最多的Pandas的对象，和Series类似，创建DataFrame时，也接受许多不同类的参数。

# ##### From dict of Series or dicts

# In[60]:


d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}


# In[61]:


df = pd.DataFrame(d)
df


# In[62]:


pd.DataFrame(d, index=['d', 'b', 'a'])


# In[64]:


pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])


# 可以通过访问索引和列属性分别访问行和列标签。

# In[43]:


df.index


# In[65]:


df.columns


# ##### From dict of ndarrays / lists

# ndarrays必须都是相同的长度。如果传递了索引，它的长度也必须与数组一样长。如果没有传递索引，结果将是range(n)，其中n是数组长度。

# In[118]:


d = {'one' : [1., 2., 3., 4.],
     'two' : [4., 3., 2., 1.]}


# In[67]:


pd.DataFrame(d)


# In[68]:


pd.DataFrame(d, index=['a', 'b', 'c', 'd'])


# ##### From structured or record array

# 这种情况和从数组的字典集合创建是一样的。

# In[69]:


np.zeros((2,))


# In[70]:


data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])


# In[71]:


data


# In[72]:


data.shape


# In[73]:


data.size


# In[74]:


data[:] = [(1,2.,'Hello'), (2,3.,"World")]
data


# In[75]:


pd.DataFrame(data, index=['first', 'second'])


# In[77]:


pd.DataFrame(data, columns=['C', 'A', 'B'])


# 注意：DataFrame和 2-dimensional NumPy ndarray 并不是完全一样的。

# 除了以上的构造方法之外还有很多其他的构造方法，但获取DataFrame的主要方法是读取表结构的文件，其他构造方法就不一一列出。

# In[112]:


get_ipython().run_line_magic('pinfo', 'df.sub')


# In[5]:


d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df
#print(df)
print (df.index)
print (df.index.values)
print (df.columns)
print (df.columns.values)
print (df.dtypes)
print (df)


# In[116]:


stockSummaries={
'AMZN': pd.Series([346.15,0.59,459,0.52,589.8,158.88],index=['Closing price','EPS','Shares Outstanding(M)','Beta', 'P/E','Market Cap(B)']),
'GOOG': pd.Series([1133.43,36.05,335.83,0.87,31.44,380.64],index=['Closing price','EPS','Shares Outstanding(M)','Beta','P/E','Market Cap(B)']),
'FB': pd.Series([61.48,0.59,2450,104.93,150.92],index=['Closing price','EPS','Shares Outstanding(M)','P/E', 'Market Cap(B)']),
'YHOO': pd.Series([34.90,1.27,1010,27.48,0.66,35.36],index=['Closing price','EPS','Shares Outstanding(M)','P/E','Beta', 'Market Cap(B)']),
'TWTR':pd.Series([65.25,-0.3,555.2,36.23],index=['Closing price','EPS','Shares Outstanding(M)','Market Cap(B)']),
'AAPL':pd.Series([501.53,40.32,892.45,12.44,447.59,0.84],index=['Closing price','EPS','Shares Outstanding(M)','P/E','Market Cap(B)','Beta'])}
stockDF=pd.DataFrame(stockSummaries)
stockDF


# In[80]:


stockDF=pd.DataFrame(stockSummaries,index=['Closing price','EPS','Shares Outstanding(M)','P/E', 'Market Cap(B)','Beta'])
stockDF


# In[81]:


stockDF=pd.DataFrame(stockSummaries,columns=['FB','TWTR','SCNW'])
stockDF


# In[ ]:


#练习：取弄一行和某一列的均值


# In[97]:


stockDF1=pd.DataFrame(stockSummaries,columns=['FB'])
#type(stockDF1)
stockDF1.mean()
#type(stockDF1.mean())
stockDF1.mean().values


# In[103]:


stockDF1=pd.DataFrame(stockSummaries,index=['Closing price'])
stockDF1
stockDF1.values
type(stockDF1.values)
stockDF1.values.mean()


# ### DataFrame列操作

# DataFrame列的选取，设置和删除列的工作原理与类似的dict操作相同。

# In[120]:


df


# In[121]:


df['one']


# In[109]:


df


# In[122]:


df['three'] = df['one'] * df['two']
df


# In[123]:


df['flag'] = df['one'] > 2
df


# DataFram的列可以像使用dict一样被删除或移出。

# In[124]:


del df['two']
df


# In[125]:


three = df.pop('three')
df


# In[126]:


three


# In[ ]:


#df.pop('three')


# In[127]:


df['foo'] = 'bar'
df


# 如果传入的是Series并且索引不完全相同，那么会默认按照索引对齐。

# In[128]:


df['one_trunc'] = df['one'][:2]
df


# 也可以插入原始的ndarrays，但其长度必须与DataFrame索引的长度相匹配。
# 
# 默认情况下，直接的赋值操作列插入到最后的位置。insert方法可用于插入列中的特定位置：

# In[129]:


df.insert(1, 'bar', df['one'])


# In[131]:


get_ipython().run_line_magic('pinfo', 'df.insert')


# In[130]:


df


# 分配列

# In[6]:


import numpy as np
import pandas as pd


# In[7]:


df_sample = pd.DataFrame({'A': range(1, 11), 'B': np.random.randn(10)})
df_sample


# In[8]:


df_sample.A


# In[73]:


df_sample.assign(ln_A = lambda x: np.log(x.A), abs_B = lambda x: np.abs(x.B))


# In[136]:


get_ipython().run_line_magic('pinfo', 'df_sample.assign')


# 需要注意的是，传入的参数是以字典类型的方式传入的。如果希望保证顺序的话，可以多次使用assign。

# In[74]:


newcol = np.log(df_sample['A'])


# In[75]:


newcol


# In[76]:


df_sample.assign(ln_A=newcol)


# ### Indexing / Selection

# 索引的基础知识如下：

# Operation | Syntax | Result
# - | - | -
# Select column | df[col] | Series
# Select row by label | df.loc[label] | Series
# Select row by integer location | df.iloc[loc] | Series
# Slice rows | df[5:10] | DataFrame
# Select rows by boolean vector | df[bool_vec] | DataFrame

# In[137]:


df


# In[139]:


df.loc['b']


# 关于索引的内容较多，后续单独讲解。

# In[151]:


# df.iloc[行位置,列位置]
#print( df.iloc[1,1])#选取第二行，第二列的值，返回的为单个值
#print (df.iloc[[0,2],:])#选取第一行及第三行的数据
#print (df.iloc[0:2,:])#选取第一行到第三行（不包含）的数据
#print (df.iloc[:,1])#选取所有记录的第er列的值，返回的为一个Series
#print (df.iloc[1,:])#选取第一行数据，返回的为一个Series


# In[176]:


df


# 注意：在Pandas 0.20版本开始就不推荐使用.ix，只推荐使用基于标签的索引.loc 和基于位置的索引.iloc 。

# ### 数据对齐和运算

# DataFrame对象之间在列和索引（行标签）之间自动数据对齐。并且，运算的结果对象是列和行标签的并集。

# In[164]:


df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df


# In[165]:


df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
df2


# In[166]:


df + df2


# In[167]:


df


# 在DataFrame和Series之间进行操作时，默认行为是使DataFrame列上的Series索引对齐，从而逐行广播。

# In[156]:


df.iloc[0]


# In[157]:


df


# In[188]:


df - df.iloc[0]


# In[190]:


#df - df.iloc[[0,1]]


# 在使用时间序列数据等一些特殊情况下，也可以以列方式进行广播：

# In[177]:


index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
df


# In[185]:


type(df.iloc[0])


# In[184]:


type(df.iloc[[0]])


# In[161]:


df.sub(df["A"],axis=0)


# In[162]:


get_ipython().run_line_magic('pinfo', 'df.sub')


# In[114]:


df.sub(df['A'], axis=0)


# 逻辑运算，与NumPy相似。

# In[191]:


df1 = pd.DataFrame({'a' : [1, 0, 1], 'b' : [0, 1, 1] }, dtype=bool)
df2 = pd.DataFrame({'a' : [0, 1, 1], 'b' : [1, 1, 0] }, dtype=bool)


# In[192]:


df1


# In[193]:


df2


# In[194]:


df1 & df2


# In[195]:


df1 | df2


# In[196]:


df1 ^ df2


# In[198]:


df1


# In[197]:


-df1  # ~df1


# 转置

# In[199]:


df[:5].T


# ### DataFrame & ufunc

# 由于DataFrame基于NumPy构建，可以直接使用大量NumPy中ufunc函数。

# In[200]:


np.exp(df)


# In[201]:


np.asarray(df)


# # 基本功能

# ## 数据IO

# In[9]:


import numpy as np
import pandas as pd


# In[10]:


df = pd.read_csv('data/macrodata.csv')


# In[14]:


df.head()
df.tail()
df.shape


# In[15]:



print(df.index)
print(df.columns)
print(df.dtypes)


# Operation | Syntax | Result
# - | - | -
# Select column | df[col] | Series
# Select row by label | df.loc[label] | Series
# Select row by integer location | df.iloc[loc] | Series
# Slice rows | df[5:10] | DataFrame
# Select rows by boolean vector | df[bool_vec] | DataFrame

# In[ ]:


df.iloc[行位置,列位置]
#print( df.iloc[1,1])#选取第二行，第二列的值，返回的为单个值
#print (df.iloc[[0,2],:])#选取第一行及第三行的数据
#print (df.iloc[0:2,:])#选取第一行到第三行（不包含）的数据
#print (df.iloc[:,1])#选取所有记录的第er列的值，返回的为一个Series
#print (df.iloc[1,:])#选取第一行数据，返回的为一个Series


# In[20]:


df[df.year==1959]


# In[26]:


#df[df.year==1959][['m1','pop']]
#df.loc[df.year==1959,['m1','pop']]
#df[['m1','pop']][df.year==1959]
#df[['m1','pop']].loc[df.year==1959]


# ### 读取CSV/文本类文件

# In[60]:


df = pd.read_csv('data/macrodata.csv')
df.head()
#df.tail()


# In[61]:


i="realint"
box=1.5*(df[i].quantile(q=0.75)-df[i].quantile(q=0.25))
upper=df[i].quantile(q=0.75)+box
downer=df[i].quantile(q=0.25)-box
#print(sum(df[i]>upper))
#print(sum(df[i]<downer))
upper_flag=df[i]>upper
downer_flag=df[i]<downer

df[i].loc[upper_flag]=upper
df[i].loc[downer_flag]=downer


# In[50]:


#错误
sum(list(upper_flag) or list(downer_flag)) 
#sum(list(downer_flag))


# In[53]:


[True,True] or [False,False,False,False]


# In[39]:


sum(upper_flag | downer_flag)


# In[41]:


sum(upper_flag)
sum(downer_flag)


# In[321]:


get_ipython().run_line_magic('pinfo', 'df.quantile')


# In[29]:


df["realint"].quantile(q=0.5)


# In[62]:


df.shape


# In[ ]:


#练习图区bikes


# In[207]:


df['qtr']=df['year']*10+df['quarter']
df.head(3)


# In[210]:


#df.mean()
df.apply(lambda x: x.max() - x.min())


# In[212]:


#df.quantile?
df.quantile()


# In[213]:


#df.mean()#计算列的平均值，参数为轴，可选值为0或1.默认为0，即按照列运算
#df.mean(0)
#df.mean(1)[1:10]
df.sum()
#df.sum(1)#计算行的和
#df.apply(lambda x: x.max() - x.min())#将一个函数应用到DataFrame的每一列，这里使用的是匿名lambda函数，与R中apply函数类似


# In[63]:


df.describe().T


# In[ ]:


#练习：有没有离群值


# In[64]:


df['year'].astype('int')
df1=df['year'].astype('int')
df1.head()


# In[65]:


df['year'].astype('int')


# In[66]:


df['quarter'].value_counts()


# In[148]:


df['year'].value_counts().shape


# In[221]:


type(df['year'].value_counts())


# In[232]:


a=pd.DataFrame(df['year'].value_counts())
#print (a)
a.reset_index(inplace=True)
a
#b=a.reset_index(inplace=False)
#print(a)
#print(b)
#a.shape
#a


# In[70]:


df = pd.read_excel('data/bonus_schedule.xls')
df
#df.Employed_Before.value_counts()


# In[337]:


df.Employed_Before.unique()


# In[69]:


df.describe(include='all').T
#df.describe().T


# In[235]:


df[df.Bonus_Percent<0.02]


# In[161]:


df[(df.Bonus_Percent<0.02) & (df['diff']>8 )]['diff']


# In[162]:


(df.Bonus_Percent<0.02) & (df['diff']>8 )


# In[163]:


df['diff'][(df.Bonus_Percent<0.02) & (df['diff']>8 )]


# In[164]:


df[df.Bonus_Percent<0.02][['Bonus_Percent','diff']]


# In[272]:


from datetime import datetime
s="28-01-2011"
str(s) 
datetime.strptime(s, '%d-%m-%Y')

num = datetime.strptime(s, "%d-%m-%Y").year
num 


# In[297]:


df


# In[298]:


#datetime.strptime(str(df['Employed_Before'].values[1])[2:11], '%Y-%m-%d').year
datetime.strptime(str(df['Employed_Before'][3])[0:9],'%Y-%m-%d').year


# In[288]:


str(df['Employed_Before'].values[1])


# In[ ]:


datetime.strptime(str(x)[2:11], '%Y-%m-%d').year


# In[60]:


get_ipython().run_line_magic('pinfo', 'datetime.strptime')


# sort_index可以以轴的标签进行排序。axis是指用于排序的轴，可选的值有0和1，默认为0即行标签（Y轴），1为按照列标签排序。 ascending是排序方式，默认为True即降序排列

# In[76]:


print (df)

print (df.sort_index(axis=0, ascending=False))
print (df.sort_index(axis=1, ascending=False))


# In[74]:



print (df.sort_values(by='diff'))

print (df.sort_values(by=['Bonus_Percent','diff'],ascending=[0,1]))


# DataFrame也提供按照指定列进行排序，可以仅指定一个列作为排序标准（以单独列名作为columns的参数），也可以进行多重排序（columns的参数为一个列名的List，列名的出现顺序决定排序中的优先级），在多重排序中ascending参数也为一个List，分别与columns中的List元素对应

# In[306]:


df = pd.read_excel('data/test11.xls', 'Sheet1', index_col=0)
df


# 要查看Series或DataFrame对象的小样本，使用head()和tail()方法。默认要显示的元素数为5，但可以传递自定义数字。

# In[85]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# ### 读取Excel文件

# ### 读取数据库数据

# In[132]:


"""
import sqlite3
import mysql.connector
import sqlalchemy
"""


# In[88]:


mysql_engine = sqlalchemy.create_engine('mysql+mysqlconnector://root:1234@localhost/world', encoding='utf-8')
sql_table = pd.read_sql('show tables', mysql_engine)


# In[89]:


sql_table


# In[90]:


sqlite_engine = sqlalchemy.create_engine('sqlite:////WorkSpace/Data/SinaNews.db', encoding='utf-8')
sina_news = pd.read_sql("SELECT * FROM sinanewslink LIMIT 100", sqlite_engine)


# In[131]:


#sina_news.tail(10)


# ### 数据输出

# In[124]:


df.to_csv("data/test.csv", encoding="utf-8")


# In[125]:


df.to_excel("data/test.xlsx", "lookatme")


# In[1]:


#sqlite_engine_test = sqlalchemy.create_engi pd.date_range('1/1/2000', periods=8)ne('sqlite:////WorkSpace/Data/TestDB.db', encoding='utf-8')
#df.to_sql("dftest", sqlite_engine_test)


# In[ ]:


pd.read_sql("SELECT * FROM dftest", sqlite_engine_test, index_col="index").head(7)


# ## 数据选取&描述

# ### 查看数据

# Pandas对象有很多可以访问原数据的属性

# In[65]:


index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
df


# In[66]:


df.shape


# In[67]:


df.index


# In[68]:


df.columns


# In[69]:


df.values


# In[70]:


df.head()


# In[71]:


df.tail(3)


# In[72]:


df.info()


# In[73]:


df.describe(percentiles=[*np.arange(0.2,1,0.2)])


# In[74]:


df.T


# In[75]:


df.sort_index(axis=1, ascending=False)


# In[76]:


df.sort_values(by='B')


# 注意：排序默认的都是返回一个新的对象。如果需要修改原数据，那么需要传入参数 inplace=True.

# ### 选择&查询

# 在进行数据的选取查询时需要注意获取到的是一个复制还是一个视图

# 获取列

# In[170]:


df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
                   'three' : pd.Series(np.random.randn(3), index=['b', 'c','d'])})
df


# In[171]:


df.one


# In[79]:


df["two"]


# 获取行

# In[172]:


df[1:3]


# In[173]:


df["a":"c"]


# In[174]:


df['four'] = ['A', 'A','B','C']
df


# In[84]:


df[df['four'].isin(['A','C'])]


# ### 基于标签的索引

# .loc是基于标签的索引，必须使用数据的标签属性，否则会返回一个异常

# Object Type | Indexers
# -|-
# Series | s.loc[indexer]
# DataFrame | df.loc[row_indexer,column_indexer]

# In[175]:


df2 = pd.DataFrame(np.random.randn(6,4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
df2


# In[86]:


df2.loc['b':'d']


# 注意：基于标签的索引切片会包含首尾的两个索引

# In[87]:


df2.loc['a']


# 可以传入单个字符串或数字索引到某行或列，但注意即使传入的是数值表示的也是数值的标签，而非位置。

# In[88]:


df2.loc[['a', 'b', 'd'], :]


# In[89]:


df2.loc['d':, 'A':'C']


# In[90]:


df2.loc[:, df2.loc['a'] > 0]


# In[91]:


df2.loc['d','B']


# In[92]:


df2.at['d','B']


# .loc同时也可以用来扩展。

# In[176]:


df2


# In[93]:


df2.loc[:,"E"] = df2.loc[:,"A"].apply(abs)
df2


# In[177]:


df2.loc["g"] = 8.88
df2


# Series也可以使用.loc操作

# In[178]:


ser = df2.B
ser


# In[179]:


ser.loc['c':]


# In[180]:


ser.loc['e']


# ### 基于位置的索引

# .iloc是基于位置的索引，传入的参数为目标区域的位置索引。

# In[181]:


ser2 = pd.Series(np.random.randn(5), index=list(range(0,10,2)))


# In[182]:


ser2


# In[183]:


ser.iloc[:3]


# In[ ]:


ser.iloc[3]


# In[ ]:


ser.iloc[:3] = 0
ser


# In[ ]:


df4 = pd.DataFrame(np.random.randn(6,4),
                   index=list(range(0,12,2)),
                   columns=list(range(0,8,2)))
df4


# In[ ]:


df4.iloc[:3]


# In[ ]:


df4.iloc[1:5, 2:4]


# In[ ]:


df4.iloc[[1, 3, 5], [1, 3]]


# In[ ]:


df4.iloc[1:3, :]


# In[ ]:


df4.iloc[:, 1:3]


# In[ ]:


df4.iloc[1, 1]


# In[ ]:


df4.iat[1,1]


# 注意：单个值（标量）索引方式 .at 和 .iat 在效率和低开销上比.loc和.iloc都要优秀很多。

# In[ ]:


df4.iloc[1]


# 使用函数作为参数

# In[98]:


df5 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
df5


# In[99]:


df5.loc[lambda df: df.A > 0, :]


# In[100]:


df5.loc[:, lambda df: ['A', 'B']]


# In[101]:


df5.iloc[:, lambda df: [0, 1]]


# In[102]:


df5[lambda df: df.columns[0]]


# In[103]:


df5.A.loc[lambda s: s > 0]


# In[106]:


df5.loc[lambda s: s.A+s.B>0,:]


# In[114]:


df5


# In[137]:


np.sum(df5.values,axis=1)


# In[121]:


get_ipython().run_line_magic('pinfo', 'df.sum')


# In[130]:


df5


# In[135]:


#df5.loc[lambda s: s.sum(axis=1)>0,:]
df5.loc[:,lambda s: s.sum(axis=0)>0]


# ### *混合索引

# 警告：从Pandas 0.20.0开始，.ix索引（混合索引）已被弃用，推荐使用更严格的.iloc和.loc索引器。

# 原有的的混合索引方式
# 
# ```
# df.ix[[0, 2], 'A']
# ```
# 
# 可以用下面的两种方式进行替代：
# 
# ```
# df.loc[df.index[[0, 2]], 'A']
# 
# df.iloc[[0, 2], df.columns.get_loc('A')]
# ```

# 如果是多个索引，可以使用 .get_indexer
# ```
# df.iloc[[0, 2], df.columns.get_indexer(['A', 'B'])]
# ```

# ### 布尔型索引

# 另一个常见的操作是使用布尔向量来过滤选取数据。 
# ```
#  | 代表 or, & 代表 and,  ~ 代表 not  
# 其中表示not的也有使用 - 的，但是建议使用 ~
# ```
# 这些运算必须使用括号分组。

# In[ ]:


s = pd.Series(range(-3, 4))
s


# In[ ]:


s[s > 0]


# In[ ]:


s[(s < -1) | (s > 0.5)]


# In[ ]:


s[~(s < 0)]


# In[ ]:


s[-(s < 0)]


# In[ ]:


df5


# 可以按照需要的条件筛选行

# In[ ]:


df5[df5.A > 0]


# 结合列表推导式或map函数可以处理更复杂的选取逻辑

# In[138]:


df6 = pd.DataFrame({'a' : ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b' : ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c' : np.random.randn(7)})
df6


# 获取A列值为 `'two' or 'three'`的数据

# In[140]:


criterion = df6['a'].map(lambda x: x.startswith('t'))
criterion
#df6[criterion]


# In[ ]:


df6[[x.startswith('t') for x in df6['a']]]


# 多重条件

# In[ ]:


df6[criterion & (df6['b'] == 'x')]


# 组合使用

# In[ ]:


df6.loc[criterion & (df6['b'] == 'x'),'b':'c']


# 和 `isin` 组合使用

# In[ ]:


s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s


# In[ ]:


s.isin([2, 4, 6])


# In[ ]:


s[s.isin([2, 4, 6])]


# 同样也可以应用在index对象上面。

# In[ ]:


s[s.index.isin([2, 4, 6])]


# 注意和下面的区别

# In[ ]:


s[[2, 4, 6]]


# DataFrame也有一个isin方法，返回同样大小的矩阵。

# In[ ]:


df = pd.DataFrame({'vals': [1, 2, 3, 4],
                   'ids': ['a', 'b', 'f', 'n'],
                   'ids2': ['a', 'n', 'c', 'n']})
df


# In[ ]:


values = ['a', 'b', 1, 3]
df.isin(values)


# 当然，大多数情况下是对特定列的数据进行匹配的。

# In[ ]:


values = {'ids': ['a', 'b'], 'vals': [1, 3]}
df.isin(values)


# 将DataFrame的isin与 any 和 all 方法结合，可以快速选择符合给定条件的数据子集。如：只选择每列都符合自己的标准的那些行：

# In[ ]:


values = {'ids': ['a', 'b'], 'ids2': ['a', 'c'], 'vals': [1, 3]}
row_mask = df.isin(values).all(1)
df[row_mask]


# In[ ]:


row_mask


# 通常我们返回的结果是原数据集的一个子集，使用where方法可以保证选择输出与原始数据的形状相同。

# In[ ]:


s[s > 0]


# In[ ]:


s.where(s > 0)


# In[ ]:


df5[df5 < 0]


# 在返回的副本中，可使用其他参数替换条件为False的值。也可以通过inplace来修改原对象。

# In[ ]:


df5.where(df5 < 0, other=-df5)


# In[ ]:


df6 = df5.copy()

df6[df6 < 0] = 0


# In[ ]:


df6


# 使用query()方法

# In[ ]:


df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))
df


# In[ ]:


df[(df.a < df.b) & (df.b < df.c)]


# In[ ]:


df.query('(a < b) & (b < c)')


# 这样的方法也可以应用到索引上

# In[ ]:


df = pd.DataFrame(np.random.randint(10 / 2, size=(10, 2)), columns=list('bc'))
df.index.name = 'a'
df


# In[ ]:


df.query('a < b and b < c')


# 如果索引没有被命名，也可以直接使用index表示索引。

# In[ ]:


df.query('index < b < c')


# 注意：如果列名称和索引名称重复，那么列会被优先使用。

# ### 随机抽样

# 使用sample()方法可以从Series，DataFrame随机选择行或列。   
# 该方法将默认采样行，并接受指定数量的行/列返回。

# In[ ]:


s = pd.Series([0,1,2,3,4,5])
s


# 默认抽样为一个

# In[ ]:


s.sample()


# In[ ]:


s.sample(n=3)


# 也可以指定返回百分比，但是不能与n同时使用。

# In[ ]:


s.sample(frac=0.5)


# 默认情况下，一行样本最多返回一次，即为无放回抽样。但也可以使用replace选项替换抽样方式：

# In[ ]:


s.sample(n=6, replace=False)


# In[ ]:


s.sample(n=6, replace=True)


# 默认情况下，每一行被选择的概率相等。但是如果希望各行具有不同的概率，则可以提供权重作为sample方法的抽样权重。  
# 权重可以是列表，数字数组或Series，但长度必须与要采样的对象长度相同。
# 
# 注意：如果权重之和不为1，那么函数会将所有权重除以权重之和来重新归一化。

# In[ ]:


example_weights = [0, 0, 0.2, 0.2, 0.2, 0.4]
s.sample(n=3, weights=example_weights)


# In[ ]:


example_weights2 = [0.5, 0, 0, 0, 0, 0]
s.sample(n=1, weights=example_weights2)


# 应用于DataFrame时，可以使用DataFrame的列作为采样权重（采样行而不是列的情况下），只需将列的名称作为字符串传递进去即可。

# In[ ]:


df2 = pd.DataFrame({'col1':[9,8,7,6], 'weight_column':[0.5, 0.4, 0.1, 0]})
df2


# In[ ]:


df2.sample(n = 3, weights = 'weight_column')


# 通过sample的axis参数可以对列进行采样

# In[ ]:


df3 = pd.DataFrame({'col1':[1,2,3], 'col2':[2,3,4]})
df3


# In[ ]:


df3.sample(n=1, axis=1)


# 如果需要重现随机结果，可以通过 random_state 参数设置随机数种子。

# In[ ]:


df4 = pd.DataFrame({'col1':[1,2,3], 'col2':[2,3,4]})
df4


# In[ ]:


df4.sample(n=2, random_state=2)


# In[ ]:


df4.sample(n=2, random_state=2)


# 关于axis参数：
# - 使用0值表示沿着每一列或行标签\索引值向下执行方法
# - 使用1值表示沿着每一行或者列标签模向执行对应的方法

# ## 缺失值处理

# Pandas中主要使用`np.nan`来表示缺失的数据。默认情况下不会被包括在计算中。

# In[307]:


df = pd.DataFrame(np.random.randn(5, 3),
                  index=['a', 'c', 'e', 'f', 'h'],
                  columns=['one', 'two', 'three'])

df['four'] = 'bar'
df['five'] = df['one'] > 0

df


# In[308]:


df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df2


# In[310]:


df3=df2.copy()
df3
print(id(df3))
id(df2)


# In[313]:


df3.loc[df3["four"].isnull(),"four"]="bar"
df3


# 为了使检测到的缺失值更容易（并且跨不同列和数据类型）。  
# Pandas提供了isnull（）和notnull（）函数，它们也是Series和DataFrame对象的方法：

# In[314]:


pd.isnull(df2['one'])


# In[315]:


df2.isnull()


# 需要注意的是在NumPy中np.nan之间是不同的。

# In[146]:


None == None


# In[147]:


np.nan == np.nan


# In[148]:


type(np.nan)


# 因此，像下面的这样的条件判断是无效的。

# In[149]:


df2['one'] == np.nan


# ### 插入缺失值

# In[166]:


df3 = df[['one', 'two', 'three']].copy()
df3
df3.iloc[[0,1,-1],0] = np.nan


# In[167]:


df3


# In[ ]:


df2 = df.copy()
df2['timestamp'] = pd.Timestamp('20120101')


# In[ ]:


df2


# In[ ]:


df2.loc[['a','c','h'],['one','timestamp']] = np.nan


# In[ ]:


df2


# In[ ]:


df2.get_dtype_counts()


# ### 缺失值的运算

# - 当进行求和运算时，缺失的数据将会被当作零来计算。
# - 如果数据全部为NA，那么结果也会返回NA。
# - 像 cumsum 和 cumprod 等方法回忽略NA值，但是会在返回的结果数组中回显示缺失的值。

# In[168]:


df3


# In[169]:


df3['one'].sum()


# In[171]:


df3.mean(0)


# In[172]:


df3.cumsum()


# GroupBy中的NA值  
# 在GroupBy中NA值会被直接忽略，这点同R相同。

# In[173]:


df3.groupby('one').mean()


# 有关详细信息，请参阅groupby部分。

# ### 清理/填补缺失数据

# Pandas对象具有多种数据处理方法来处理缺失的数据。

# fillna方法可以通过几种方式将非空数据填补到NA值的位置。

# ##### 使用标量填补

# In[316]:


df2


# In[317]:


df2.fillna(0)


# In[ ]:


df2['four'].fillna('missing')


# ##### 向前填充或向后填充

# In[183]:


df2


# In[182]:


df2.fillna(method='bfill',inplace=True)


# In[180]:


get_ipython().run_line_magic('pinfo', 'df2.fillna')


# In[ ]:


df2.fillna(method='bfill', limit=1)


# 其中填补的方式：
# <table>
# <colgroup>
# <col width="38%">
# <col width="63%">
# </colgroup>
# <thead valign="bottom">
# <tr class="row-odd"><th class="head">Method</th>
# <th class="head">Action</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr class="row-even"><td>pad / ffill</td>
# <td>Fill values forward</td>
# </tr>
# <tr class="row-odd"><td>bfill / backfill</td>
# <td>Fill values backward</td>
# </tr>
# </tbody>
# </table>

# ##### 使用Pandas对象填充

# 也可以使用可对齐的dict或Series进行填充。Series的index 或 dict的标签必须与要填充的框架的列匹配。

# In[ ]:


dff = pd.DataFrame(np.random.randn(10,3), columns=list('ABC'))
dff.iloc[3:5,0] = np.nan
dff.iloc[4:6,1] = np.nan
dff.iloc[5:8,2] = np.nan
dff


# In[ ]:


dff.fillna(dff.mean())


# In[ ]:


dff.fillna(dff.mean()['B':'C'])


# In[ ]:


dff.where(pd.notnull(dff), dff.mean(), axis='columns')


# ### 删除缺失值

# In[ ]:


df3["one"] = np.nan
df3


# In[ ]:


df3.dropna(axis=0)


# In[ ]:


df3.dropna(axis=1)


# In[ ]:


df3['one'].dropna()


# 在删除缺失值时需要注意的是，DataFrame有两条轴，而Series只有一条，所以需要指定在哪条轴上操作。

# ### 插值

# Series和Dataframe对象都具有插值方法，默认情况下，在缺少的数据点处执行线性插值。

# In[ ]:


ser = pd.Series([0.469112, np.nan, 5.689738, np.nan, 8.916232])
ser


# In[ ]:


ser.interpolate()


# In[ ]:


df = pd.DataFrame({'A': [1, 2.1, np.nan, 4.7, 5.6, 6.8],
                   'B': [.25, np.nan, np.nan, 4, 12.2, 14.4]})


# In[ ]:


df


# In[ ]:


df.interpolate()


# 其中可以通过参数来控制插值的方式，不同的参数对应不同的差值方式和插值算法。详细内容见scipy。

# - liner: 忽略索引对缺失部分进行等距的线性插值
# - time: 在时间索引下，按日期或者更高的时间频率按等距进行插值。
# - index, values: 使用索引的实际数值。
# - 其他参数: 参见scipy内容。

# In[ ]:


df.interpolate(method='index')


# In[ ]:


df.interpolate(method='akima')


# 当通过多项式或样条近似进行插值时，还需要传入order参数。

# In[ ]:


df.interpolate(method='polynomial', order=2)


# 同样可以通过limit参数来限制插值的个数。

# In[ ]:


ser = pd.Series([np.nan, np.nan, 5, np.nan, np.nan, np.nan, 13])
ser


# In[ ]:


ser.interpolate(limit=2)  # limit_direction == 'forward'


# In[ ]:


ser.interpolate(limit=1, limit_direction='backward')


# In[ ]:


ser.interpolate(limit=1, limit_direction='both')


# ### 替换

# 对于Series，可以通过另一个值替换单个值或多个值，也可以使用列表对应替换，或者使用字典指定替换多个值，也可以使用填补的方式来进行替换。

# In[318]:


ser = pd.Series([0., 1., 2., 3., 4.])

ser.replace(0, 5)


# In[ ]:


ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])


# In[ ]:


ser.replace({0: 10, 1: 100})


# In[ ]:


ser.replace([1, 2, 3], method='pad')


# 对于数据框可以通过列来替换单个值

# In[ ]:


df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]})
df


# In[ ]:


df.replace({'a': 0, 'b': 5}, 100)


# 字符串/正则表达式替换

# In[ ]:


d = {'a': list(range(4)), 'b': list('ab..'), 'c': ['a', 'b', np.nan, 'd']}

df = pd.DataFrame(d)
df


# 字符串替换：

# In[ ]:


df.replace('.', np.nan)


# 正则表达式替换

# In[ ]:


df.replace(r'\s*\.\s*', np.nan, regex=True)


# 列表对应的字符串替换

# In[ ]:


df.replace(['a', '.'], ['b', np.nan])


# 列表对应的正则表达式替换

# In[ ]:


df.replace([r'\.', r'(a)'], ['dot', '\1stuff'], regex=True)


# # 分组运算-GroupBy

# 通过使用GroupBy可以做到涉及一个或多个以下步骤的过程：
# 
# - 根据一些标准将数据拆分成组
# - 独立应用功能给每个组
# - 将结果合并成数据结构

# ## 将对象拆分成组

# In[194]:


df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})


# In[195]:


df


# 可以使用一个或多个键执行GroupBy，键可以使用列或者行。

# In[198]:


grouped = df.groupby('A')
#grouped = df.groupby(['A', 'B'])
grouped.size()


# In[192]:


def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'


grouped = df.groupby(get_letter_type, axis=1)


# In[193]:


grouped.groups


# In[190]:


df


# 默认情况下，在groupby操作期间会进行排序。可以通过sort = False来进行潜在的加速：

# In[ ]:


df3 = pd.DataFrame({'X' : ['A', 'B', 'A', 'B'], 'Y' : [1, 4, 3, 2]})
df3


# In[ ]:


df3.groupby(['X']).get_group('A')


# In[ ]:


df3.groupby(['X']).get_group('B')


# ## GroupBy对象属性

# In[ ]:


df.groupby('A').groups


# In[ ]:


grouped = df.groupby(['A', 'B'])
grouped.groups


# 查看分组个数

# In[ ]:


len(grouped)


# 一旦创建了GroupBy对象，有几种方法可用于对分组数据执行计算

# In[ ]:


grouped = df.groupby('A')
grouped.aggregate(np.sum)


# In[ ]:


grouped = df.groupby(['A', 'B'])
grouped.aggregate(np.sum)


# 默认键值会被作为新的多层的索引，可以使用as_index=False来控制作为列输出，或者在生成的对象上使用reset_index()

# In[ ]:


grouped.size() #查看分组大小


# 同时应用多个函数

# In[ ]:


grouped = df.groupby('A')


# In[ ]:


grouped['C'].agg([np.sum, np.mean, np.std])


# 在DataFrame的分组上，传递一个应用于每个列的函数列表，该列产生具有分层索引的聚合结果：

# In[ ]:


grouped.agg([np.sum, np.mean, np.std])


# 如果需要对生成的列重新命名

# In[ ]:


grouped['C'].agg([np.sum, np.mean, np.std]).rename(columns={'sum': 'foo',
                                                            'mean': 'bar',
                                                            'std': 'baz'})


# In[ ]:


grouped.agg([np.sum, np.mean, np.std]).rename(columns={'sum': 'foo','mean': 'bar','std': 'baz'})


# 对不同的列使用不同的函数

# In[ ]:


grouped.agg({'C' : np.sum,
             'D' : lambda x: np.std(x, ddof=1)})


# # 补充

# In[199]:


uefaDF=pd.read_csv('data/euro_winners.csv')
uefaDF.head()


# In[202]:


uefaDF.shape
uefaDF.describe(include="all")


# In[205]:


uefaDF['Nation'].value_counts()


# In[5]:


uefaDF['Nation'].value_counts().index


# In[ ]:


tuple(nationsGrp.groups)


# In[6]:


nationindex=uefaDF['Nation'].value_counts().index
nationindex
int(len(nationindex))


# In[7]:


name=uefaDF['Nation'].value_counts().index
results=[]
for i in range(len(name)):
    flag=uefaDF['Nation']==name[i]
    tmp=uefaDF.Attendance[flag].mean()
    results.append(tmp)
    
results


# In[207]:


nationsGrp =uefaDF.groupby('Nation');
type(nationsGrp)


# In[208]:


nationsGrp.groups


# In[10]:


d=nationsGrp.groups
for key in d:
    value=d[key]
    print (value)
    #print uefaDF.iloc[x,].mean()


# In[210]:


ddd=nationsGrp.groups
for key in ddd:
    v=ddd[key]
   # print (v,key)
    print (uefaDF.iloc[v,].mean())
    


# In[12]:


ddd=nationsGrp.groups
ddd.keys()


# In[13]:


ddd=nationsGrp.groups
ddd.values()


# In[211]:


nationsGrp.mean()


# In[212]:


nationsGrp.size()


# In[213]:


nationWins=nationsGrp.size()
print (nationWins.sort_values(ascending=True,inplace=False))
#nationWins


# In[17]:


nationWins.sort_values(ascending=True,inplace=True)
nationWins


# In[223]:


uefaDF["num"]=uefaDF["Attendance"]*1.5


# In[224]:


winnersGrp =uefaDF.groupby(['Nation','Winners'])
clubWins=winnersGrp.size()
clubWins
#clubWins.sort_values(ascending=False)
#clubWins


# In[228]:


winnersGrp.describe(include="all").T


# In[217]:


winnersGrp.groups


# In[238]:


goalStatsDF=pd.read_csv('data/goal_stats_euro_leagues_2012-13.csv')
#goalStatsDF=goalStatsDF.set_index('Month')
goalStatsDF


# In[233]:


goalStatsDF.index


# In[236]:


goalStatsDF=goalStatsDF.reset_index()
goalStatsDF
m=goalStatsDF['Month']
year=lambda x: x.split('/')

year(m[2])


# In[22]:


m=goalStatsDF['Month']
year=lambda x: x.split('/')[2]
#year(m[1])
goalStatsDF['Year']=m.apply(year,1)
goalStatsDF.head()


# In[239]:


y=goalStatsDF['Month']
year=lambda x : x.split('/')[2]
years=[]
for i in range(len(y)):
    years.append(year(y[i]))
goalStatsDF['Year']=years
    
goalStatsDF


# In[240]:


goalStatsDF=pd.read_csv('data/goal_stats_euro_leagues_2012-13.csv')
goalStatsDF['Year']=goalStatsDF['Month'].apply(lambda x: x.split('/')[2],1)
goalStatsDF=goalStatsDF.set_index('Month')
goalStatsDF.head()


# In[241]:


YearsGrp =goalStatsDF.groupby('Year')
YearsGrp.groups


# In[242]:


goalStatsDF.head(3)
goalStatsDF.tail(3)


# In[243]:


for name, group in YearsGrp:
    print (name)
    print (group)


# In[244]:


goalStatsGroupedByMonth = goalStatsDF.groupby(level=0)
for name, group in goalStatsGroupedByMonth:
    print (name)
    print (group)
    print ("\n")


# In[250]:


goalStatsDF=goalStatsDF.reset_index()
#goalStatsDF
goalStatsDF=goalStatsDF.set_index(['Month','Stat'])
goalStatsDF
monthStatGroup=goalStatsDF.groupby(level=['Month','Stat'])
#for name, group in monthStatGroup:
    #print (name)
    #print (group)


# In[252]:


goalStatsDF2=pd.read_csv('data/goal_stats_euro_leagues_2012-13.csv')
goalStatsDF2=goalStatsDF2.set_index(['Month','Stat'])
print (goalStatsDF2.head(3))


# In[253]:


grouped2=goalStatsDF2.groupby(level='Stat')
grouped2.sum()


# In[254]:


goalStatsDF2.sum(level='Stat')


# In[255]:


totalsDF=grouped2.sum()
totalsDF.loc['GoalsScored']/totalsDF.loc['MatchesPlayed']


# In[259]:


gpg=totalsDF.loc['GoalsScored']/totalsDF.loc['MatchesPlayed']
gpg
goalsPerGameDF=pd.DataFrame(gpg).T
goalsPerGameDF


# In[260]:


goalsPerGameDF=goalsPerGameDF.rename(index={0:'GoalsPerGame'})
goalsPerGameDF


# In[48]:


#pd.options.display.float_format='{:.2f}'.format
totalsDF.append(goalsPerGameDF)


# In[261]:


#grouped2=goalStatsDF2.groupby(level='Stat')
grouped2.sum()


# In[262]:


grouped2.aggregate(np.sum)


# In[263]:


grouped2.agg([np.sum, np.mean,np.size])


# In[264]:


goalStatsDF2=pd.read_csv('data/goal_stats_euro_leagues_2012-13.csv')
goalStatsDF2


# In[265]:


uefaDF=pd.read_csv('data/euro_winners.csv')
uefaDF.head()


# In[267]:


uefaDF.describe(include="all")


# In[ ]:


uefaDF["Venue"].replace(, ['dot', '\1stuff'], regex=True)


# In[268]:


get_ipython().run_line_magic('pinfo', 'df.replace')


# In[279]:


#uefaDF.to_csv('data/euro_winners1111.csv')
uefaDF111=pd.read_csv('data/euro_winners1111.csv',encoding="GBK")
uefaDF111.head()


# In[274]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[340]:


dd=pd.read_csv('data/hz_weather.csv')
dd.describe(include="all").T


# In[351]:


results=dd.风力.value_counts().reset_index()
results["温度均值差"]=np.nan
results


# In[359]:


dd1=dd.astype("object")
dd1.info()


# In[360]:


dd.info()

