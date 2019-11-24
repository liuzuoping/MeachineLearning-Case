import os
import pandas as pd

os.chdir(r'D:\Python_book\9DT')
#pd.set_option('display.max_columns', None)


# In[2]:


data = pd.read_csv('AllElectronics.csv', skipinitialspace=True)
data.head()


# In[3]:


target = data['buys_computer']
data = data.ix[:, 'age':'credit_rating']
data.head()


# ## CART算法(分类树)
# 建立CART模型

# In[14]:


import sklearn.tree as tree

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=12345)  # 当前支持计算信息增益和GINI
clf.fit(data, target)


# In[15]:


#get_ipython().magic('pinfo tree.DecisionTreeClassifier')


# In[16]:


tree.export_graphviz(clf, out_file='cart.dot')


# 可以使用graphviz将树结构输出，在python中嵌入graphviz可参考：[pygraphviz](http://www.blogjava.net/huaoguo/archive/2012/12/21/393307.html)

# # 可视化

# 使用dot文件进行决策树可视化需要安装一些工具：
# - 第一步是安装graphviz。linux可以用apt-get或者yum的方法安装。如果是windows，就在官网下载msi文件安装。
#    无论是linux还是windows，装完后都要设置环境变量，将graphviz的bin目录加到PATH，
#    比如windows，将C:/Program Files (x86)/Graphviz2.38/bin/加入了PATH
# - 第二步是安装python插件graphviz： pip install graphviz
# - 第三步是安装python插件pydotplus: pip install pydotplus

# In[11]:


import pydotplus
from IPython.display import Image
import sklearn.tree as tree


# In[18]:


dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
#%%

