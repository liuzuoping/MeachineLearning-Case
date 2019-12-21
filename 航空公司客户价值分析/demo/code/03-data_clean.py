#-*- coding: utf-8 -*- 

# 代码7-6

# 处理缺失值与异常值

import numpy as np
import pandas as pd

datafile = '../data/air_data.csv'  # 航空原始数据路径
cleanedfile = '../tmp/data_cleaned.csv'  # 数据清洗后保存的文件路径

# 读取数据
airline_data = pd.read_csv(datafile,encoding = 'utf-8')
print('原始数据的形状为：',airline_data.shape)

# 去除票价为空的记录
airline_notnull = airline_data.loc[airline_data['SUM_YR_1'].notnull() & 
                                   airline_data['SUM_YR_2'].notnull(),:]
print('删除缺失记录后数据的形状为：',airline_notnull.shape)

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录。
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM']> 0) & (airline_notnull['avg_discount'] != 0)
index4 = airline_notnull['AGE'] > 100  # 去除年龄大于100的记录
airline = airline_notnull[(index1 | index2) & index3 & ~index4]
print('数据清洗后数据的形状为：',airline.shape)

airline.to_csv(cleanedfile)  # 保存清洗后的数据
