import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


'''
打开数据 17 列，分成三类 

	                                	                                                               
绝对不能当特征	: cnt、casual、registered、instant、dteday	                                     |目标本身；目标的分解；行号；字符串日期
这一步先用	:          temp、atemp、hum、windspeed	                                            |4 个真正的连续量
先放着,P4 再处理 :	  season、yr、mnth、hr、holiday、weekday、workingday、weathersit	         | 存成了整数，但含义是类别：hr=23 不是「23 倍」，weathersit=4 不是「4 倍坏」。硬当数字用会引入假的大小关系
'''
df = pd.read_csv('hour.csv')
'''
 dteday是日期，不能作为特征值,casual,registered和cnt是用户数，也不能做特征值，
 instant虽然相关系数是0.28，但是用instant会导致泛化能力变弱，

 temp,atemp,hum,windspeed。温度，体感温度，湿度以及风速确实比较重要，是比较重要的影响共享单车租借的因素,而且他们的数值有对应的物理意义，适合线性模型计算
 
 至于season,yr,mnth,hr,hoilday,weekday,workingday、weathersit这些，
 season,weathersit这两个的数字是分类编号，所以不能粗略的当成数字计算，
 hr与weekday的数字具有周期性，比如23的下一个其实是0，但是纯计算会把0和23差23个数来计算，会产生很大的误差
'''

keep_colunms = ['temp','atemp','hum','windspeed'] #只保留本次需要的四个特征值
X = df[keep_colunms]
y = df['cnt']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#random_state=42是为了确保每次切分的数据都一样

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
常数基线：不做任何训练，全部测试样本都预测训练集的平均值。
预测平均值时的 RMSE，数学上就等于数据的标准差，所以这两个数应该非常接近。

'''
y_mean = y_train.mean()
print('y训练集的平均值为:',y_mean)
y_pred1 = np.full(len(y_test),y_mean)
y_squared = mean_squared_error(y_test,y_pred1)
y_sqrt = np.sqrt(y_squared)
print(y_squared)
print(y_sqrt)
print(y_test.std())
print('===========================')

#单特征线性回归：只用 temp

model = LinearRegression() 
model.fit(X_train[['temp']],y_train)        #注意特征写法是 X_train[['temp']]
y_pred2 = model.predict(X_test[['temp']])
y_squared2 = mean_squared_error(y_test,y_pred2)
y_sqrt2 = np.sqrt(y_squared2)
print(y_squared2)
print(y_sqrt2)
print(y_test.std())
print('==========================')



'''
 四个特征
把 temp、atemp、hum、windspeed 全用上，重复上面的流程。记下 RMSE。
'''

model = LinearRegression() 
model.fit(X_train[['temp','atemp','hum','windspeed']],y_train)
y_pred3 = model.predict(X_test[['temp','atemp','hum','windspeed']])
y_squared3 = mean_squared_error(y_test,y_pred3)
y_sqrt3 = np.sqrt(y_squared3)
print(y_squared3)
print(y_sqrt3)
print(y_test.std())
print('==========================')


#这次不切分：在全部数据上 fit，再在同一批数据上预测、算 RMSE。但是一般不会做这一步，只是方便对比学习



model = LinearRegression() 
model.fit(X[['temp','atemp','hum','windspeed']],y)
y_pred4 = model.predict(X_test[['temp','atemp','hum','windspeed']])
y_squared4 = mean_squared_error(y_test,y_pred4)
y_sqrt4 = np.sqrt(y_squared4)
print(y_squared4)
print(y_sqrt4)
print(y_test.std())


'''
全部数据fit的rmse比4特征的rmse小了0.07，原因是数据很多但是参数很少，所以说模型根本就没有过拟合的能力
但是数据肯定不是正确的，如果预测新的一天误差应该在150左右
因为评估集不对，所以说模型会去迎合数据的局部特征
'''






