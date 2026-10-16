import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold, cross_val_score,TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline




#先看一眼共线性

df = pd.read_csv('hour.csv')

X = df[['temp','atemp','windspeed','hum']]
y = df['cnt']


print(X['temp'].corr(X['atemp'])) #相关系数是0.98.说明很相关了，说明多重共线性
plt.figure(1,figsize=(14,7))
plt.scatter(X['temp'],X['atemp'],alpha = 0.5)
plt.title('collinearity_scatter')
plt.xlabel('temp')
plt.ylabel('atemp')
plt.savefig('figures/collinearity_scatter.png')


'''
 线性回归给它们的系数是多少
必须先标准化再 fit，否则系数不可比。用 StandardScaler 对 4 个特征标准化，然后 LinearRegression().fit(Xs, y)，打印 .coef_。
'''
scaler = StandardScaler()
Xs = scaler.fit_transform(X)   #X = df[['temp','atemp','windspeed','hum']]
model = LinearRegression() 
model.fit(Xs,y)
print('四个特征值的系数为：',model.coef_)
'''[ 16.4778042   54.01834594   5.25794063    -53.08896458]
   'temp'       'atemp'    'windspeed'     'hum'

差不多是三倍，但是不能说体感温度比温度重要三倍，因为他们俩的相关系数是0.98，强相关，如果temp稍微变了一点，atemp也会变一点，但是分配的系数却相差三倍，这是很容易出问题的
'''

'''

 系数到底稳不稳
用 KFold(5, shuffle=True, random_state=42)，每个折单独 fit 一个线性回归，把系数收集起来，看同一个特征在不同折上的跨度（最大值 - 最小值）。
要打印成一张表：每个特征一行，列出 5 个折的系数 + 跨度。

这里的 KFold 只是诊断系数稳定性，不是评估模型，所以用随机 KFold 没问题，但要清楚它和主协议不是一回事。

'''
kf = KFold(5, shuffle=True, random_state=42)
train_coef_ = []
for train_index,test_index in kf.split(Xs):
    X_train = Xs[train_index]
    y_train = y[train_index]
    model = LinearRegression() 
    model.fit(X_train,y_train)
    train_coef_.append(model.coef_)

Data_coef_ = pd.DataFrame(train_coef_,columns=['temp','atemp','windspeed','hum'])
Data_coef_ = Data_coef_.T

Data_coef_['span'] = Data_coef_.max(axis= 1)-Data_coef_.min(axis = 1)
print('===5次测试的系数为===')
print(Data_coef_)
# temp和atemp的跨度比较大，，剩下俩的跨度很小，差了八九倍甚至十几倍

'''
Ridge：扫 alpha

① 用 cross_val_score + 你的主协议 拿 RMSE
② 再在全量上 fit 一次，读 pipe.named_steps['ridge'].coef_

记录两样东西：每个 alpha 的 ① RMSE ② 四个系数。

'''

coef_list = []
rmse_list = []
tscv = TimeSeriesSplit(n_splits=5)
for a in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=a))
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring='neg_root_mean_squared_error')
    rmse_list.append(-scores.mean())
    pipe.fit(X,y)
    coef = pipe.named_steps['ridge'].coef_
    coef_list.append(coef)
pd_rmse = pd.DataFrame(rmse_list)
pd_coef = pd.DataFrame(coef_list,columns=['temp','atemp','windspeed','hum'])
print('===7种alpha情况下的rmse值===')
print(pd_rmse)
print('===7种alpha情况下的四个特征值的参数===')
print(pd_coef)


alpha = [0.01, 0.1, 1, 10, 100, 1000, 10000]
plt.figure(2,figsize=(14,7))

for feature in ['temp', 'atemp', 'windspeed', 'hum']:
    plt.semilogx(alpha, pd_coef[feature], marker='o', label = feature)
plt.xlabel('Alpha (Log Scale)', fontsize=12)
plt.ylabel('Coefficient Value', fontsize=12)
plt.title('Ridge Coefficient Path', fontsize=14)
plt.legend(fontsize=12)
plt.savefig('figures/ridge_coef_path.png')


plt.figure(3,figsize=(14,7))
plt.semilogx(alpha, rmse_list, marker='o')
plt.xlabel('Alpha',fontsize=12)
plt.ylabel('rmse',fontsize=12)
plt.title('ridge_rmse')
plt.savefig('figures/ridge_rmse.png')


'''
明显看出是alpha = 100的时候，rmse最低，但是也就比168.94到168.48，其实就下降了0.46
temp和atemp是聚拢的。temp先上升后下降，atemp一直下降
alpha过大的时候，rmse为177多，模型欠拟合，
系数上，atemp与temp相等，然后windspeed几乎不变，hum绝对值减小

'''

'''
用 Ridge 去救 P4 的失控
回头用 degree 8 的多项式特征(494 列)


用同一个时间留出（前 80% 训练 / 后 20% 测试），扫几个 alpha（0.01 ~ 100）。


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
    
df_rmse = pd.DataFrame(result)
print('===degree从0到8时rmse的值===')
print(df_rmse)



coef_list1 = []
rmse_list1 = []
result1 = []
n = int(len(df) * 0.8)
X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]
alpha = [0.01, 0.1, 1, 10, 100]
for a in alpha :
    pipe = make_pipeline(PolynomialFeatures(8, include_bias=False),
                     StandardScaler(),
                     Ridge(alpha=a))
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_list1.append(rmse)
    coef = pipe.named_steps['ridge'].coef_
    coef_list1.append(coef)
df_rmse1 = pd.DataFrame(rmse_list1,index = alpha,columns = ['RMSE'])
df_coef1 = pd.DataFrame(coef_list1,index = alpha)
print('===Ridge之后rmse值===')
print(df_rmse1)
print('===Ridge之后特征值的系数===')
print(df_coef1)
plt.show()


# 494特征的 比只用 4 个特征的线性回归（209.54）好了大概3%，因此没有必要为了那点提升，把特征从 4 个扩到 494 个















