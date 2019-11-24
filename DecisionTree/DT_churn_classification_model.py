import pandas as pd
import numpy as np

import os
os.chdir(r"D:\Python_book\9DT")
# In[2]:


churn = pd.read_csv('telecom_churn.csv')  # 读取已经整理好的数据
churn.head()


# In[3]:

# In[4]:
import matplotlib.pyplot as plt
import seaborn as sns

# In[5]:
sns.barplot(x='edu_class', y='churn',data=churn)
plt.show()


# In[8]:


sns.boxplot(x='churn', y='peakMinDiff', hue=None, data=churn)
plt.show()


# In[10]:


sns.boxplot(x='churn', y='duration', hue='edu_class', data=churn)
plt.show()


# ##筛选变量
# * 筛选变量时可以应用专业知识，选取与目标字段相关性较高的字段用于建模，也可通过分析现有数据，用统计量辅助选择
# * 为了增强模型稳定性，自变量之间最好相互独立，可运用统计方法选择要排除的变量或进行变量聚类

# In[11]:


corrmatrix = churn.corr(method='spearman')  # spearman相关系数矩阵，可选pearson相关系数，目前仅支持这两种,函数自动排除category类型
corrmatrix_new=corrmatrix[np.abs(corrmatrix) > 0.5]  # 选取相关系数绝对值大于0.5的变量，仅为了方便查看
#  为了增强模型稳定，根据上述相关性矩阵，可排除'posTrend','planChange','nrProm','curPlan'几个变量


# In[12]:
#连续型变量往往是模型不稳定的原因;
#如果所有的连续变量都分箱了,可以统一使用卡方检验进行变量重要性检验
churn['duration_bins'] = pd.qcut(churn.duration,5)  #  将duration字段切分为数量（大致）相等的5段
churn['churn'].astype('int64').groupby(churn['duration_bins']).agg(['count', 'mean'])


# In[14]:


bins = [0, 4, 8, 12, 22, 73]
churn['duration_bins'] = pd.cut(churn['duration'], bins, labels=False)
churn['churn'].astype('int64').groupby(churn['duration_bins']).agg(['mean', 'count'])


# In[15]:


#根据卡方值选择与目标关联较大的分类变量
#计算卡方值需要应用到sklearn模块，但该模块当前版本不支持pandas的category类型变量，会出现警告信息，可忽略该警告或将变量转换为int类型
import sklearn.feature_selection as feature_selection

churn['gender'] = churn['gender'].astype('int')
churn['edu_class'] = churn['edu_class'].astype('int')
churn['feton'] = churn['feton'].astype('int')
feature_selection.chi2(churn[['gender', 'edu_class', 'feton', 'prom', 
                              'posPlanChange','duration_bins', 'curPlan', 'call_10086']], churn['churn'])#选取部分字段进行卡方检验
#根据结果显示，'prom'、'posPlanChange'、'curPlan'字段可以考虑排除


# ## 建模
# * 根据数据分析结果选取建模所需字段，同时抽取一定数量的记录作为建模数据
# * 将建模数据划分为训练集和测试集
# * 选择模型进行建模

# In[16]:


#  根据模型不同，对自变量类型的要求也不同，为了示例，本模型仅引入'AGE'这一个连续型变量
#model_data = churn[['subscriberID','churn','gender','edu_class','feton','duration_bins']]
model_data = churn[['subscriberID','churn','gender','edu_class','feton','duration_bins','call_10086','AGE']]#第二可选方案
model_data.head()


# In[17]:


import sklearn.model_selection as cross_validation

target = model_data['churn']  # 选取目标变量
data=model_data.ix[:, 'gender':]  # 选取自变量

train_data, test_data, train_target, test_target = cross_validation.train_test_split(data,target, test_size=0.4, train_size=0.6 ,random_state=12345) # 划分训练集和测试集


# In[18]:


# 选择决策树进行建模
import sklearn.tree as tree

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target)  #  使用训练数据建模

# 查看模型预测结果
train_est = clf.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clf.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=clf.predict(test_data)  #  用模型预测测试集的结果
test_est_p=clf.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率
pd.DataFrame({'test_target':test_target,'test_est':test_est,'test_est_p':test_est_p}).T # 查看测试集预测结果与真实结果对比


# ## 模型评估

# In[19]:


import sklearn.metrics as metrics

print(metrics.confusion_matrix(test_target, test_est,labels=[0,1]))  # 混淆矩阵
print(metrics.classification_report(test_target, test_est))  # 计算评估指标
print(pd.DataFrame(list(zip(data.columns, clf.feature_importances_))))  # 变量重要性指标


# In[20]:

#察看预测值的分布情况
red, blue = sns.color_palette("Set1", 2)
sns.distplot(test_est_p[test_target == 1], kde=False, bins=15, color=red)
sns.distplot(test_est_p[test_target == 0], kde=False, bins=15,color=blue)
plt.show()


# In[21]:


fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color=blue)
plt.plot(fpr_train, tpr_train, color=red)
plt.title('ROC curve')
plt.show()
#这里表现出了过渡拟合的情况
#########################################################################################
#参数调优
#%%
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[2,3,4,5,6,7,8],
    'min_samples_split':[4,8,12,16,20,24,28] 
}
clf = tree.DecisionTreeClassifier()
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, 
                   scoring='roc_auc', cv=4)
clfcv.fit(train_data, train_target)
#%%
# 查看模型预测结果
train_est = clfcv.predict(train_data)  #  用模型预测训练集的结果
train_est_p=clfcv.predict_proba(train_data)[:,1]  #用模型预测训练集的概率
test_est=clfcv.predict(test_data)  #  用模型预测测试集的结果
test_est_p=clfcv.predict_proba(test_data)[:,1]  #  用模型预测测试集的概率
#%%
fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)
plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, color=blue)
plt.plot(fpr_train, tpr_train, color=red)
plt.title('ROC curve')
plt.show()
#%%
clfcv.best_params_

#%%
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=24) # 当前支持计算信息增益和GINI
clf.fit(train_data, train_target)  #  使用训练数据建模
#%%
# ### 可视化

# 使用dot文件进行决策树可视化需要安装一些工具：
# 
# - 第一步是安装[graphviz](http://www.graphviz.org/)。linux可以用apt-get或者yum的方法安装。如果是windows，就在官网下载msi文件安装。无论是linux还是windows，装完后都要设置环境变量，将graphviz的bin目录加到PATH，比如windows，将C:/Program Files (x86)/Graphviz2.38/bin/加入了PATH
# - 第二步是安装python插件graphviz： pip install graphviz
# - 第三步是安装python插件pydotplus: pip install pydotplus

# In[39]:


import pydotplus
from IPython.display import Image
import sklearn.tree as tree


# In[18]:


dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=train_data.columns,
    max_depth=5,
    class_names=['0','1'],
    filled=True
) 
            
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
#%%

"""
# ## 模型保存/读取

# In[36]:


import pickle as pickle

model_file = open(r'clf.model', 'wb')
pickle.dump(clf, model_file)
model_file.close()


# In[37]:


model_load_file = open(r'clf.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()

test_est_load = model_load.predict(test_data)
pd.crosstab(test_est_load,test_est)
#%%
"""
