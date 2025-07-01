
# coding: utf-8

# # Numpy介绍
# 
#          

# NumPy 是 Numerical Python 的简称，是高性能计算和数据分析的基础包。
# 
# 虽然Python是用于通用编程的优秀工具，具有高度可读的语法，丰富而强大的数据类型（字符串，列表，集合，字典，数字等）以及非常全面的标准库，
# 
# 特别是，Python列表是非常灵活的容器，可以任意深度嵌套，并且可以容纳任何Python对象，但它并不是专为数学和科学计算而设计，并不不适合有效地表示常用
# 
# 的数学结构（向量和矩阵）。 
# 
# 其语言和标准库中都没有用于多维数据集高效表示的工具，线性代数工具和一般矩阵操作（实际上是所有技术计算的基本组成部分）。
# 
# 那么是否能够精通理解数组的思维，掌握面向数组的编程，将会为你的Python生涯奠定坚实的根基。其实Numpy本身并没有很多的高级数据分析的功能。
# 
# 本章所介绍的 NumPy 基础主要是作为未来学习和使用 Pandas 包进行数据分析, 所以不会提及太复杂的内容。首先对比列表，认识Nupmy数组；随后进阶掌握
# 
# ndarray，学习Numpy的高效运算。

# ## 1. 对比列表认识数组

# ### （1）创建一个列表和一个一维数组

# In[1]:


import numpy as np


# In[14]:


list1=[5,10,15,20]
type(list1)
list1


# In[15]:


arr1=np.array([5,10,15,20])
print (arr1)
type(arr1)


# In[16]:


import sys
from datetime import datetime
import numpy as np


# In[6]:


def pythonsum(n):
    a = list(range(n))
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i *2
        b[i] = i *3
        c.append(a[i] + b[i])
    return c

pythonsum(6)


# In[7]:


list(range(6))*2


# In[8]:


np.arange(6)*2


# In[10]:


def numpysum(n):
       a = np.arange(n) * 2
        
       b = np.arange(n) * 3
    
       c = a + b

       return c

numpysum(6)


# In[44]:


size = input('input a num ')
size=int(size)
start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print ("The last 2 elements of the sum", c[-2:])
print ("PythonSum elapsed time in microseconds", delta.microseconds)


# In[45]:


size = input('input a num ')
size=int(size)
start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print ("The last 2 elements of the sum", c[-2:])
print ("NumPySum elapsed time in microseconds", delta.microseconds)


# ### （2）对比列表与数组的索引

# In[7]:


list1[0]


# In[9]:


arr1[0]


# In[10]:


list1[:2]


# In[17]:


arr1[:2]


# ### （3）列表与数组的不同

# 列表和数组之间的第一个区别是数组是同类的; 即数组的所有元素必须具有相同的类型。相反，列表可以包含任意类型的元素。例如，我们可以将上面列表中的最后一个元素更改为一个字符串。

# In[10]:


list1[-1]="CDA"
list1


# In[11]:


arr1[-1]="CDA"


# ### 2.ndarray

# ### 通过与列表的对比，我们已经初步认识了Numpy数组的创建方法，Numpy数组的属性索引以及数据类型。
# ### 实际上NumPy数组是一个多维数组对象，称之为ndarray。关于NumPy数组首先需要掌握的就是：
# ### 1）NumPy数组的下标从0开始。
# ### 2）同一个NumPy数组中所有元素的类型必须是相同的。
# 

# ### （1）在正式深度探索之前，我们首先对数组的基础知识有个初步的认识。
# NumPy数组的维数称为秩（rank），一维数组的秩为1，二维数组的秩为2，以此类推。在NumPy中，每一个线性的数组称为是一个轴（axes），秩其实是描述轴的数量。比如说，二维数组相当于是两个一维数组，其中第一个一维数组中每个元素又是一个一维数组。所以一维数组就是NumPy中的轴（axes），第一个轴相当于是底层数组，第二个轴是底层数组里的数组。而轴的数量—秩，就是数组的维数。

# ### NumPy的数组中比较重要的属性函数有：
# ndarray.ndim：数组的维数（即数组轴的个数），等于秩。最常见的为二维数组（矩阵）。
# ndarray.shape：数组的维度。为一个表示数组在每个维度上大小的整数元组。例如二维数组中，表示数组的“行数”和“列数”。ndarray.shape返回一个元组，这个元组的长度就是维度的数目，即ndim属性。
# ndarray.size：数组元素的总个数，等于shape属性中元组元素的乘积。
# ndarray.dtype：表示数组中元素类型的对象，可使用标准的Python类型创建或指定dtype。另外也可以查看数组的数据类型。
# ndarray.itemsize：数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为8(float64占用64个bits，每个字节长度为8，所以64/8，占用8个字节），又如，一个元素类型为complex32的数组item属性为4（32/8）。
# ndarray.data：包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。

# ### Numpy中的基本数据类型

# bool	用一个字节存储的布尔类型（True或False）
# int8	一个字节大小，-128 至 127
# int16	整数，-32768 至 32767
# int32	整数，-2 ** 31 至 2 ** 32 -1
# int64	整数，-2 ** 63 至 2 ** 63 - 1
# uint8	无符号整数，0 至 255
# uint16	无符号整数，0 至 65535
# uint32	无符号整数，0 至 2 ** 32 - 1
# uint64	无符号整数，0 至 2 ** 64 - 1
# float16	半精度浮点数：16位，正负号1位，指数5位，精度10位
# float32	单精度浮点数：32位，正负号1位，指数8位，精度23位
# float64或float	双精度浮点数：64位，正负号1位，指数11位，精度52位
# complex64	复数，分别用两个32位浮点数表示实部和虚部
# complex128或complex	复数，分别用两个64位浮点数表示实部和虚部

# ### NumPy数据类型的转换

# In[20]:


arr1


# ### （2）数组的创建

# ### 数组的创建方式有很多种，本节课我们主要讲解常见的创建数组的方法。
# 

# ### 第一种就是我们在开篇用到的创建数组的方式：使用array函数从常规的 list， tuple等格式的数据转创建为ndarray， 默认创建一个新的数组，所创建的数组类型由原序列中的元素类型推导而来。　　　

# In[22]:


lis2=[1.1,1.2,1.3,1.4,1.5]
arr2=np.array(lis2)
arr2


# In[24]:


lst3=[[1,2,3,4],[5,6,7,8]]
arr3=np.array(lst3)
arr3
#arr3.shape


# ### 可使用双重序列来表示二维的数组，三重序列表示三维数组，以此类推。

# In[25]:


lst3=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
arr3=np.array(lst3)
arr3


# ### 第二种方法的应用场景是很多时候我们对所创建的数组内的元素未知，但数组的大小已知。针对这种情况，NumPy提供了一些使用占位符创建数组的函数，也就是说一个常量值初始化一个数组，比如：用函数zeros可创建一个全是0的数组，用函数ones可创建一个全为1的数组，函数empty创建一个内容随机并且依赖与内存状态的数组（默认数组类型(dtype)为float64）。这些函数有助于满足除了数组扩展的需要，同时降低了高昂的运算开销。

# In[27]:


print(np.zeros(5,dtype=int))
np.zeros(5,dtype=int)


# In[17]:


print(np.zeros(5,dtype=complex))


# In[29]:


np.zeros((2,3),int)


# In[30]:


np.ones((2, 3))


# In[31]:


arr2


# In[24]:


np.ones_like(arr2)


# In[25]:


np.zeros_like(arr2)


# In[32]:


np.linspace(0, 1, 12,endpoint=False,retstep=True) 
#linspace函数通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值:


# In[33]:


np.linspace(0, 1, 12,endpoint=False,retstep=False)


# In[27]:


np.linspace(2.0, 3.0, num=5, retstep=True)


# ### 那么如果我们想要用任意值创建一个初始化的数组，该如何操作？
# 
# ### 我们可以创建一个空数组，然后使用fill方法将所需的值放入数组中：

# In[36]:


a=np.empty(5)
a.fill(5.5)
a


# In[30]:


np.empty((2,3)) #创建空数组, 类似 np.ones. 其中每个元素都没有进行初始化, 并不能保证都是 0.


# In[37]:


arr2


# In[38]:


np.empty_like(arr2)


# In[31]:


np.eye(4, 3) #创建一个对角线为 1 其他为 0 的矩阵.


# In[32]:


np.identity(3)   #创建一个主对角线为 1 其他为 0 的方阵.


# ### 第三种常见的创建数组的方法.我们可以使用 arange方法，创建数组序列。

# In[39]:


b=np.arange(5)
b


# In[40]:


c=np.arange(10,30,5)
c


# ### 第四种创建随机数组
# 最后，创建具有随机数字的数组通常是非常有用的。 np.random模块包含许多可用于此效果的函数，例如，这将生成一个从标准正态分布（0均值和方差1）中抽取的5个随机样本数组：

# In[41]:


d=np.random.randn(7)
d


# In[42]:


e=np.random.normal(10,3,(2,4))
e


# In[43]:


get_ipython().run_line_magic('pinfo', 'np.random.normal')


# ### （3）数组的索引\切片与遍历

# ### 上面我们看到了如何对一维数组进行索引切片，就像Python列表一样。 ：

# In[44]:


arr=np.arange(10)
arr


# In[38]:


arr[2]


# In[45]:


arr[2:5]


# In[47]:


np.arange(24)


# In[52]:


b = np.arange(24).reshape(6,4)
b


# In[57]:


b[:,1]


# In[49]:


b.ravel()


# In[51]:


b.flatten()


# ### 多维数组

# In[58]:


def func1(x):
    return x*2
arr=np.fromfunction(func1,(5,))
arr


# In[60]:


def f(x,y):
    return 10*x+y


# In[68]:


arr2=np.fromfunction(f,(5,4),dtype=int)
arr2


# In[69]:


arr2[2,3]


# In[70]:


arr2[-2]


# In[71]:


print(arr2[0:3,0:2])


# In[72]:


arr2[1:6]


# In[73]:


arr2


# In[74]:


arr2[:2,1:]


# ### 如果只想遍历整个array可以直接使用

# In[77]:


arr3=np.random.normal(10,3,[2,4])
arr3


# In[78]:


print(arr3[1,2:3])
print(arr3[:,2])


# In[80]:


lst=[[1,2,3],[4,5,6]]
print(lst)
[i[2] for i in lst ]


# In[82]:


arr4=arr3.reshape(4,2)
arr4


# In[83]:


for row in arr4:
    print(row)


# In[85]:


for i in arr4:
    print(i)


# In[86]:


arr3d = np.arange(1,13).reshape((2,2,3))
arr3d


# # 布尔型索引

# In[87]:


word = np.array(list('abcabcdeba'))
word


# In[90]:


word == 'a'
#sum(word == 'a')


# In[91]:


from numpy.random import randn
data = randn(10,4)
data


# In[93]:


data[word == 'a']


# In[94]:


data[word == 'a', 2:]


# In[95]:


data[word == 'a', 3]


# In[96]:


word != 'a'


# In[72]:


~(word == 'a')


# In[97]:


mark = (word == 'a') | (word == 'b')
mark


# In[98]:


data[mark]


# In[75]:


(word == 'a') & (word == 'b')


# In[101]:


data[data>0]=0


# In[102]:


data[data > 0] = 0
data


# # 花式索引

# In[109]:


arr = np.empty((7,5))
for i in range(7):
    arr[i] = i
arr


# In[104]:


arr[[5, 3, 2, 1]]


# In[105]:


arr[[-1, -3, -5, 1]]


# In[111]:


arr = np.arange(32).reshape((8,4))
arr


# In[107]:


arr[[1, 5, 7 ,2], [0, 2, 1, 3]]


# In[112]:


arr[[1, 5, 7 ,2]]


# In[108]:


arr[[1, 5, 7 ,2]][:,[0, 2, 1, 3]]


# In[83]:


arr[np.ix_([1, 5, 7 ,2], [0, 2, 1, 3])]


# In[84]:


arr[[1, 5, 7 ,2], [0, 2, 1, 3]] = 0
arr


# ### （4）数组的变换
# 实际上，只要元素的总数不变，数组的形状就可以随时改变。 例如，如果我们想要一个数字从0增加的2x4数组，最简单的方法是：

# In[85]:


arr=np.random.normal(8,2,(2,6))
arr


# In[86]:


arr.reshape(3,4)


# ### 数组的另一个广泛使用的属性是.T属性数组的转置。

# In[113]:


arr=np.arange(8)

arr


# In[114]:


arr2=arr.reshape(2,4)#view
arr2


# In[130]:


arr[0]=1000
print(arr)
print(arr2)


# In[131]:


arr3=arr.copy()#copy
arr3


# In[132]:


arr[0]=5
print(arr)
print(arr2)
print(arr3)


# In[133]:


arr2.T


# ### （5）数组的其他属性

# In[134]:


arr4=arr2#view
arr4


# In[135]:


arr4.dtype


# In[136]:


arr4.size


# In[137]:


arr4.ndim


# In[138]:


arr4.shape


# In[99]:


arr4.nbytes


# In[139]:


print(arr4.min())


# In[140]:


print(arr4.max())


# In[141]:


print(arr4.sum())


# In[142]:


print(arr4.mean())


# In[143]:


print(arr4.std())


# In[144]:


arr4


# In[145]:


print(arr4.sum(axis=1))#by row


# In[146]:


print(arr4.sum(axis=0))#by column


# ### （6）数组的运算

# 数组支持所有常规算术运算符，而numpy库也包含一组完整的基本数学函数，这些函数在数组上运算。 重要的是要记住，一般来说，数组的所有操作都是以元素的形式应用的，即，同时应用于数组的所有元素。 考虑例如：

# In[147]:


arr1=np.arange(4)
arr1


# In[148]:


arr2=np.arange(10,14)
arr2


# In[149]:


print(arr1,"+",arr2,"=",arr1+arr2)


# 重要的是，你必须记住，即使乘法运算符是默认应用于元素的方式，它不是线性代数的矩阵乘法：

# In[150]:


print(arr1,"*",arr2,"=",arr1*arr2)


# 我们也可以用一个标量乘一个数组：

# In[151]:


print(arr1,"*",1.5,"=",arr1*1.5)


# In[152]:


arr = np.arange(1, 11)
arr


# In[153]:


np.sqrt(arr)


# In[155]:


np.exp(arr)


# In[157]:


x = randn(8)
y = randn(8)
print (x)
print (y)
np.maximum(x, y)


# In[160]:


arr = randn(5,4)
print(arr)
arr.mean()


# In[161]:


np.mean(arr)


# In[162]:


arr.mean(axis=1)


# In[163]:


arr


# In[165]:


arr.sum()


# In[168]:


arr.sum(0)
#arr.sum(1)


# In[169]:


arr = np.arange(9).reshape((3,3))
arr


# In[170]:


arr.cumsum(0)
arr.cumprod(1)


# In[177]:


a = np.arange(9).reshape(3,3)
a


# In[178]:


b = 2 * a
b


# In[179]:


np.hstack((a, b))


# In[180]:


np.vstack((a, b))


# In[181]:


np.column_stack((a, b))
np.row_stack((a, b))


# In[120]:


np.column_stack((a, b))==np.hstack((a, b))


# ### 数组的广播
# numpy数组的广播功能强大。
# 
# 广播原则：
# 如果两个数组的后缘维度(即：从末尾开始算起的维度)的轴长相符或其中一方的长度为1，则认为它们是广播兼容的，广播会在缺失和(或)长度为1的轴上进行.

# In[118]:


print(np.arange(3))


# In[119]:


print(np.arange(3)+5)


# In[125]:


np.ones((3,3))


# In[124]:


np.ones((3,3))+ np.arange(3)


# In[127]:


np.arange(3).reshape((3,1))+np.arange(3)


# 解析来我们用图解的方式，理解下

# ### 更强大的数学功能
# 正如我们之前提到的那样，Numpy提供了完整的数学函数，可以在整个数组上运行，包括对数，指数，三角函数和双曲三角函数等。此外，scipy还在scipy.special模块中提供了一个丰富的特殊函数库 贝塞尔，艾里，菲涅耳，拉盖尔等古典特殊功能。 例如，对0和2π之间的100点的正弦函数进行采样就像下面这样简单：

# In[130]:


x=np.linspace(0,2*np.pi,10)
y=np.sin(x)
print(x)
print(y)
print(x,y)


# ### 线性代数（NumPy）
# Numpy提供了一个基本的线性代数库，所有的数组都有一个点方法，当它的参数是向量（一维数组）时，其行为是标量点积的行为，而当它的一个或两个参数是两个时，为传统的矩阵乘法：

# ### 对于矩阵乘法，必须满足相同的维度匹配规则，例如， 考虑A×AT的区别

# In[121]:


A=randn(5,4)
A


# In[122]:


A.T


# and $A^T \times A$:

# In[123]:


print(np.dot(A.T,A))


# In[124]:


print(np.dot(A,A.T))


# 此外，numpy.linalg模块还包括附加功能，如行列式，矩阵范数，Cholesky，特征值和奇异值分解等。对于更线性的代数工具，scipy.linalg包含经典LAPACK库中的大部分工具 作为在稀疏矩阵上运行的函数。 如果大家想要了解更多关于这些内容，可以去Numpy和Scipy文档中去拓展学习。

# 
# 

# # ## 唯一化和集合逻辑

# In[182]:


names = np.array(['Atom', 'Lucy', 'Kid', 'Atom', 'Kid', 'Atom'])
names


# In[184]:


np.unique(names)


# In[183]:


ints = np.array([1,2,3,4,2,4,3,5])
ints


# In[123]:


np.unique(ints)


# In[128]:


val_1 = np.array([4,5,6,2,5,7,4,0])
val_2 = np.array([1,2,3,4,7,8,4,3])


# In[129]:


np.in1d(val_1, [0, 5, 7])


# In[126]:


np.intersect1d(val_1, val_2)


# In[130]:


np.union1d(val_1, val_2)


# In[131]:


np.setdiff1d(val_1, val_2)


# In[132]:


val_1.sort()
val_1


# In[135]:


arr = randn(5, 3)
arr.sort(1)
arr


# In[135]:


arr.sort(0)
arr


# # 补充
