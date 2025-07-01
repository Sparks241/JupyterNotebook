#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



# In[2]:
current_path=os.path.abspath(__file__)
data_path=current_path+"/data/taobao_data_matplolib.csv"
df = pd.read_csv(data_path,encoding='gbk')
df.head()


# In[12]:


Volume=np.array(df.成交量)
a2=np.array(df.价格)
Volume


# In[19]:


d=np.round(list(Volume/sum(Volume)),2)
d


# # 画出各省份价格、各省份成交量柱状图：

# In[ ]:





# # 画出成交量线图、柱状图、箱线图、饼图

# In[ ]:





# # 画出价格与成交量的散点图

# In[ ]:





# 
