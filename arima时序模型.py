#-*- coding: utf-8 -*-
#arima时序模型
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf

# 参数初始化
discfile = '../data/arima_data.xls'

# 读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile,index_col=0)
print(data.head())
print('\n Data Types:')
print(data.dtypes)


# 时序图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
data.plot()
plt.show()


#自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data).show()


#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore


#差分后的时序图
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']
D_data.plot() #时序图
plt.show()


#自相关图
plot_acf(D_data).show()


#偏自相关图
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data).show() 


#平稳性检测
print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分'])) 


#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
#返回统计量和p值 


# 一阶差分
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = data.diff(1)
diff1.plot(ax=ax1)


# 二阶差分
fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
diff2 = data.diff(2)
diff2.plot(ax=ax2)


# 合适的p,q
dta = data.diff(1)[1:]
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(dta[u'销量'],lags=10,ax=ax1)
ax2 = fig.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(dta[u'销量'],lags=10,ax=ax2)


#模型
arma_mod20 = sm.tsa.ARMA(dta,(2,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod01 = sm.tsa.ARMA(dta,(0,1)).fit()
print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
arma_mod10 = sm.tsa.ARMA(dta,(1,0)).fit()
print(arma_mod10.aic,arma_mod10.bic,arma_mod10.hqic)


#残差QQ图
resid = arma_mod01.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)


#残差自相关检验
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_mod01.resid.values.squeeze(), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_mod01.resid, lags=10, ax=ax2)


#D-W检验
print(sm.stats.durbin_watson(arma_mod01.resid.values))


# Ljung-Box检验
import numpy as np
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
datap = np.c_[range(1,36), r[1:], q, p]
table = pd.DataFrame(datap, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


#预测
predict_sunspots = arma_mod01.predict('2015-2-07', '2015-2-15', dynamic=True)
fig, ax = plt.subplots(figsize=(12, 8))
print(predict_sunspots)
predict_sunspots[0] += data['2015-02-06':][u'销量']
data=pd.DataFrame(data)
for i in range(len(predict_sunspots)-1):
    predict_sunspots[i+1]=predict_sunspots[i]+predict_sunspots[i+1]
print(predict_sunspots)
ax = data.ix['2015':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()
