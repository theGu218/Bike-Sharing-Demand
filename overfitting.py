import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold, cross_val_score,TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('hour.csv')

'''
【模型评估协议】
① 特征集合：temp, atemp, hum, windspeed
② 切分方式：TimeSeriesSplit(n_splits=5)   ← 主协议
   理由：本项目要回答"用过去预测未来"，随机切分违反 IID 前提。
   对照：随机切分 / KFold(shuffle) 只用来量化"泄漏能压低多少"，不作主协议。
③ 指标：RMSE
④ 适用范围：用过去的数据预测未来半年的小时租借量，取决于预测的是未来哪一段。因此不适用于精细化的实时车辆调度。

'''


y = df['cnt']
def evaluate_model(X, y):

 n = int(len(df) * 0.8)
 X_train, X_test = X.iloc[:n], X.iloc[n:]
 y_train, y_test = y.iloc[:n], y.iloc[n:]

 model = LinearRegression() 
 model.fit(X_train,y_train)
 y_pred = model.predict(X_test)
 y_squared = mean_squared_error(y_test,y_pred)
 y_sqrt = np.sqrt(y_squared)
 return y_sqrt
 
X_a = df[['temp','atemp','hum','windspeed']] 
print('A的rmse为：',evaluate_model(X_a, y)) #这个测的是不含hr特征值的rmse

X_b = df[['temp','atemp','hum','windspeed','hr']]
print('B的rmse为：',evaluate_model(X_b, y))  #这个是只加hr特征值的rmse

hr_dummies = pd.get_dummies(df['hr'], prefix='hr', dtype=float)
print(hr_dummies.head())
print(hr_dummies.shape)
X_c =  pd.concat([df[['temp','atemp','hum','windspeed']],hr_dummies], axis=1)
print(X_c.head())
print('C的rmse为：',evaluate_model(X_c, y))  #这个是讲hr独热的rmse，最大意义在于给上午8点和下午17点这两个高峰期分配比较大的权重

#因为发现hr具有周期性，23之后是0，所以尝试利用周期的正余弦函数看看分析分析rmse
hr_sin = np.sin(2*np.pi*df['hr']/24)
hr_cos = np.cos(2*np.pi*df['hr']/24)

X_cycle = pd.DataFrame({'hr_sin': hr_sin,'hr_cos': hr_cos})
print(X_cycle.head())
X_d = pd.concat([X_a,X_cycle],axis = 1)
print(X_d.head())
print('D的rmse为：',evaluate_model(X_d, y))


'''
多项式阶数扫描（过拟合曲线）
只在这 4 个数值特征上（temp, atemp, hum, windspeed）加多项式，看复杂度上升会怎样：

用同一个时间留出：前 80% 训练 / 后 20% 测试

每个 degree 记两个数：训练 RMSE（在自己训练的那 80% 上）和留出 RMSE（在那 20% 上）。

'''

result = []
def model_fit(X_train,X_test,y_train,y_test):
    
    model = LinearRegression() 
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
    y_rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_train))
    return {
       'degree' :d,
       'features':Xp.shape[1],
       'rmse_train':y_rmse_train,
       'rmse_test':y_rmse_test
       }
    

for d in [1, 2, 3, 4, 5, 6, 8]:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    Xp = pf.fit_transform(df[['temp','atemp','hum','windspeed']])
    n = int(len(df) * 0.8)
    X_train, X_test = Xp[:n], Xp[n:]
    y_train, y_test = y.iloc[:n], y.iloc[n:]
    result.append(model_fit(X_train,X_test,y_train,y_test))
    result_rmse = result

    
df_rmse = pd.DataFrame(result_rmse)
print(df_rmse)

'''
把两条线画出来
横轴：多项式阶数；纵轴：RMSE
两条线：训练误差 + 留出误差，带 legend
一个没有正则化的线性回归，遇到几乎共线的一堆特征，模型会出现过拟合，系数会变得很夸张，导致分配的权重过高，出现rmse乱跳的情况
'''
d = df_rmse['degree']            
y_train_curve = df_rmse['rmse_train']  
y_test_curve = df_rmse['rmse_test']  
  
plt.figure(1,figsize=(14,7))
plt.title('overfitting_curve')
plt.xlabel('degree')
plt.ylabel('RMSE')
plt.xticks(d)
plt.plot(d,y_train_curve,marker='o', color='blue', label='RMSE (Train)')
plt.plot(d,y_test_curve,marker='s', color='red', label='RMSE (Test)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.savefig('figures/overfitting_curve.png')


'''
对照：换个切分方式再看一遍
协议不变 —— 这只是 P3 里说好的"对照实验"。

把任务 2 的循环再跑一遍，但评估换成：

随机切分（train_test_split(random_state=42)）
KFold(n_splits=5, shuffle=True) 的均值


随机单次留出的那几个数（degree 3~8）跳动得厉害吗？这反过来印证了 P3 的哪一条结论？
单次随机切分的评估是极其不可靠的，偶然性极大，必须用交叉验证或时间切分来获得稳定的评估。

'''

result_rmse = []
for d in [1, 2, 3, 4, 5, 6, 8]:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    Xp = pf.fit_transform(df[['temp','atemp','hum','windspeed']])
    X_train,X_test,y_train,y_test = train_test_split(Xp, y, test_size=0.2, random_state=42)
    result_rmse.append(model_fit(X_train,X_test,y_train,y_test))
df_rmse2 = pd.DataFrame(result_rmse)
print(df_rmse2)

result = []
for d in [1, 2, 3, 4, 5, 6, 8]:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    Xp = pf.fit_transform(df[['temp','atemp','hum','windspeed']])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_train_rmse_list = []
    kf_test_rmse_list = []
    for train_index, test_index in kf.split(Xp):
        
        X_train, X_test = Xp[train_index], Xp[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
        
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
       
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        
        kf_train_rmse_list.append(rmse_train)
        kf_test_rmse_list.append(rmse_test)
    kf_train_rmse = np.mean(kf_train_rmse_list)
    kf_test_rmse = np.mean(kf_test_rmse_list)
    result.append({
        'degree': d,
        'features': Xp.shape[1],
        'kf_train_rmse': kf_train_rmse,
        'kf_test_rmse': kf_test_rmse
    })
df_kf_rmse = pd.DataFrame(result)
print(df_kf_rmse)

'''
复杂度低时，学不到东西，属于欠拟合
复杂度过高时，模型学会了太多东西，过拟合
'''



models = ['no hr', 'hr', 'get_dummies_hr']
rmse_values = [209.5, 200.3, 157.9] 

plt.figure(2,figsize=(8, 4))
plt.barh(models, rmse_values, height=0.5)
for i, v in enumerate(rmse_values):
    plt.text(v + 2, i, f"{v:.2f}", va='center', fontsize=12)
plt.title('RMSE Comparison (Lower is Better)', fontsize=14)
plt.xlabel('RMSE')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('figures/hr_encoding_comparison.png')
plt.show()

  
    
    




  
    
    



