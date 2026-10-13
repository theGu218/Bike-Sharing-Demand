import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('hour.csv')



print(df.shape)       
print(df.columns)
print(df.head())
print(df.describe())     
print(df.isna().sum())   #查缺失值
print('===========================')

'''
temp 那一列的值会在 0 到 1 之间 —— 它不是摄氏度，是被归一化过的。
Readme 给了反推参数：t_min = -8、t_max = 39。所以：
真实摄氏度 = temp * (39 - (-8)) + (-8)
'''
df['temp'] = df['temp'] * (39 - (-8)) + (-8)
print(df.head(10))
print('=============================')


hr_cnt_mean = df.groupby('hr')['cnt'].mean()
plt.figure(figsize=(10,6))
plt.bar(hr_cnt_mean.index,hr_cnt_mean.values)
plt.xlabel('hour(0-23h)')
plt.ylabel('cnt_mean')
plt.title('Bike rentals by hour')
plt.savefig('figures/Bike rentals by hour.png')
plt.show()

#两个高峰，分别在8点和18点，是上下班的高峰期，说明使用自行车的频率和时间关系很大

print('================================')

print(df.corr(numeric_only=True)['cnt'].sort_values())
#相关性最强的三个是temp，atemp和hum，说明影响是否骑车最重要的因素是温度，体感温度和湿度
print('==============================')


if(df['cnt'] == df['casual']+df['registered']).all():
    print('完全相等')
else:
    print('不相等')
#没有意义，因为模型不应该知道这些数据，模型应该做的是预测数据，这属于数据泄露了，没有任何意义

print('===================================')


corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('heatmap')
plt.savefig('figures/heatmap.png')
plt.show()
#利用seaborn画一个热图更加直观



