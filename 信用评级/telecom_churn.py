import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', None)
os.chdir(r"D:\Python_book\8Logistic")

# 导入数据和数据清洗

# In[5]:

churn = pd.read_csv(r'telecom_churn.csv', skipinitialspace=True)
churn.head()


# ##  分类变量的相关关系

# 交叉表

# In[6]:

cross_table = pd.crosstab(churn.posTrend, 
                         churn.churn, margins=True)
cross_table


# 列联表

# In[7]:

def percConvert(ser):
    return ser/float(ser[-1])

cross_table.apply(percConvert, axis=1)


# In[8]:

print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))


# ## 逻辑回归

# In[9]:

churn.plot(x='duration', y='churn', kind='scatter')


# •随机抽样，建立训练集与测试集

# In[10]:

train = churn.sample(frac=0.7, random_state=1234).copy()
test = churn[~ churn.index.isin(train.index)].copy()
print(' 训练集样本量: %i \n 测试集样本量: %i' %(len(train), len(test)))


# In[11]:

lg = smf.glm('churn ~ duration', data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary()

# 预测

# In[19]:

train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)

test['proba'].head()

# In[12]:
# ## 模型评估
# 
# 设定阈值

# In[20]:

test['prediction'] = (test['proba'] > 0.3).astype('int')


# 混淆矩阵

# In[22]:

pd.crosstab(test.churn, test.prediction, margins=True)


# - 计算准确率

# In[23]:

acc = sum(test['prediction'] == test['churn']) /np.float(len(test))
print('The accurancy is %.2f' %acc)


# In[25]:

for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.churn,
                                   margins = True)
    precision = confusion_matrix.ix[0, 0] /confusion_matrix.ix['All', 0]
    recall = confusion_matrix.ix[0, 0] / confusion_matrix.ix[0, 'All']
    Specificity = confusion_matrix.ix[1, 1] /confusion_matrix.ix[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))


# - 绘制ROC曲线

# In[27]:

import sklearn.metrics as metrics

fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()


# In[28]:

print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))

# 包含分类预测变量的逻辑回归
#%%
formula = '''churn ~ C(avgplan)'''

lg_m = smf.glm(formula=formula, data=train, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg_m.summary()


# In[14]:
#- 多元逻辑回归
# 向前法
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)


# In[16]:

candidates = ['churn','duration','AGE','edu_class','incomeCode','feton','peakMinAv','peakMinDiff','call_10086']
data_for_select = train[candidates]

lg_m1 = forward_select(data=data_for_select, response='churn')
lg_m1.summary()


# Seemingly wrong when using 'statsmmodels.stats.outliers_influence.variance_inflation_factor'

# In[17]:

def vif(df, col_i):
    from statsmodels.formula.api import ols
    
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)


# In[18]:

exog = train[candidates].drop(['churn'], axis=1)

for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))

#%%





