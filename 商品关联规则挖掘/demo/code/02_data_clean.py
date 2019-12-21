# -*- coding: utf-8 -*-


import pandas as pd
inputfile='../data/GoodsOrder.csv'
data = pd.read_csv(inputfile,encoding = 'gbk')

# 根据id对“Goods”列合并，并使用“，”将各商品隔开
data['Goods'] = data['Goods'].apply(lambda x:','+x)
data = data.groupby('id').sum().reset_index()

# 对合并的商品列转换数据格式
data['Goods'] = data['Goods'].apply(lambda x :[x[1:]])
data_list = list(data['Goods'])

# 分割商品名为每个元素
data_translation = []
for i in data_list:
    p = i[0].split(',')
    data_translation.append(p)
print('数据转换结果的前5个元素：\n', data_translation[0:5])
