

import numpy as np
import pandas as pd

inputfile = '../data/data.csv' # 输入的数据文件
data = pd.read_csv(inputfile) # 读取数据

# 描述性统计分析
description = [data.min(), data.max(), data.mean(), data.std()]  # 依次计算最小值、最大值、均值、标准差
description = pd.DataFrame(description, index = ['Min', 'Max', 'Mean', 'STD']).T  # 将结果存入数据框
print('描述性统计结果：\n',np.round(description, 2))  # 保留两位小数



# 代码6-2

# 相关性分析
corr = data.corr(method = 'pearson')  # 计算相关系数矩阵
print('相关系数矩阵为：\n',np.round(corr, 2))  # 保留两位小数



# 代码6-3

# 绘制热力图
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False
plt.subplots(figsize=(10, 10)) # 设置画面大小 
sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Blues") 
plt.title('相关性热力图')
plt.show()
plt.close



