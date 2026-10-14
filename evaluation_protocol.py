import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold, cross_val_score,TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

'''
【评估协议 —— P4~P8 沿用，不许改】
① 特征集合：temp, atemp, hum, windspeed
② 切分方式：TimeSeriesSplit(n_splits=5)   ← 主协议
   理由：本项目要回答"用过去预测未来"，随机切分违反 IID 前提。
   对照：随机切分 / KFold(shuffle) 只用来量化"泄漏能压低多少"，不作主协议。
③ 指标：RMSE（scoring='neg_root_mean_squared_error'，记得先取负）
④ 适用范围：用过去的数据预测未来半年的小时租借量，RMSE 落在 124~208，
   取决于预测的是未来哪一段。因此不适用于精细化的实时车辆调度。

【参考结果 —— 会随模型变，不属于协议】
   同分布随机切分（含泄漏，乐观）     153.1
   5 折 KFold shuffle（同样乐观）   156.8
   单次时间留出（后 20%，最难）      209.5
   TimeSeriesSplit(5)              168.9（各折 124 ~ 208）

'''





'''
三种方法，三个假设
方法	                              隐含假设	                           什么时候不成立
留出法（train/test split）	       测试集和训练集同分布	                 数据有分布漂移
k 折交叉验证	                   上面的 + 样本之间可以互换（IID）	      样本之间有顺序/分组关系
时间切分 / TimeSeriesSplit	       用过去预测未来	                     几乎没有 —— 这是最贴近真实部署的假设


'''

'''

任务 1 · 先写下你的预测（别跳过这一步）
在看任何数字之前，用一句话写下你的判断：

「三种切分方式跑同一个模型，我猜它们的 RMSE 排序是 ______ ， 因为 ______」

先把这句写进注释，再往下做。 预测错了比预测对了更有价值 —— 那说明你的直觉里有东西要修。


我猜RMSE排序是时间切分>k折>留出，因为时间切分是几乎最贴近真实部署的假设，然后k折交叉比留出法更全面一点吧，但是谨记没有免费的午餐

'''

'''
任务 2 · 随机切分（复现 P2 的数字）
train_test_split(X, y, test_size=0.2, random_state=42)，四特征线性回归，算 RMSE。

应该得到和 P2 一样的数（random_state 固定了它）。

'''

df = pd.read_csv('hour.csv')

keep_columns = ['temp','atemp','hum','windspeed'] #只保留本次需要的四个特征值
X = df[keep_columns]
y = df['cnt']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression() 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_squared = mean_squared_error(y_test,y_pred)
y_sqrt = np.sqrt(y_squared)
print(y_squared)
print(y_sqrt)
print(y_test.std())
print('==========================')

'''
任务 3 · 按时间切分（前 80% 训练、后 20% 测试）
关键：不能打乱，直接按行顺序切。

n = int(len(df) * 0.8)
X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]

先别急着下结论。 记下这个 RMSE，然后回答：

它比随机切分高还是低？
和你任务 1 的预测一致吗？

'''
n = int(len(df) * 0.8)
X_train, X_test = X.iloc[:n], X.iloc[n:]
y_train, y_test = y.iloc[:n], y.iloc[n:]

model = LinearRegression() 
model.fit(X_train,y_train)
y_pred2 = model.predict(X_test)
y_squared2 = mean_squared_error(y_test,y_pred2)
y_sqrt2 = np.sqrt(y_squared2)
print(y_squared2)
print(y_sqrt2)
print(y_test.std())
print('==========================')

#这个数据大的多，甚至比随机分配要大的多得多，至于为什么会这样，应该是因为留出法本身的弊端，就是留出法他只对训练的这一部分数据预测准确，但是至于预测未来会变得非常垃圾

'''
任务 4 · 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LinearRegression(), X, y, cv=kf,
                         scoring='neg_root_mean_squared_error')

注意两个坑：

scoring 用的是 neg_root_mean_squared_error —— 回归任务里 sklearn 返回负数（越大越好 ≈ 误差越小），所以记得 -scores
打印出5 个分数、均值、标准差
回答：这 5 个分数之间的差距大吗？

'''

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LinearRegression(), X, y, cv=kf,
                         scoring='neg_root_mean_squared_error')
print(-scores)
print(-scores.mean())  #这个就是rmse
print(scores.std())

#差距其实蛮小的，但是这也暴露了一些问题，就是模型没有算时间等其他特征，所以这个数据只是盲目乐观


#现在你手上有：随机 153、按时间切分那个数。先别解释，先取证。

print(df['dteday'].iloc[n], '~', df['dteday'].iloc[-1])     # 测试段是哪段时间
print(y.iloc[:n].mean(), y.iloc[n:].mean())                 # 训练段 / 测试段 的目标均值
print(y.iloc[:n].std(),  y.iloc[n:].std())                  # 两段的目标波动
print(df['season'].iloc[:n].value_counts().sort_index())    # 训练段的季节分布
print(df['season'].iloc[n:].value_counts().sort_index())    # 测试段的季节分布

'''
回答（这三问是本步的全部价值）：

1.测试段是哪段时间？训练段覆盖了哪些季节，测试段覆盖了哪些？
时间是2012-08-07 ~ 2012-12-31，训练段覆盖了四种季节，但是测试段只有三种，没有2季节


2.训练段和测试段的目标均值差多少？ 这意味着模型会遇到什么？
174.63914263108683-248.75460299194475 = -74，这意味着模型的误差太大，因为模型一直在以174均值去训练，但是测试的均值是248

3.所以「按时间切分的 RMSE 更高」这件事，有几个原因？分别是什么？ （提示：至少两个 —— 一个和"泄漏"有关，另一个和"未来和过去不一样"有关）
因为数据泄露？？随机切分的 153 被泄漏压低了 —— 模型训练时见过同一时段相邻的小时
还有一个应该是按时间划分的，没有考虑季节天气等特征，导致目标波动太大，就是问题2里模型会遇到的情况，模型以低均值去训练但是却去测试高均值，误差很大
时间切分撞上了真实的分布漂移 

'''


# 任务 6 · 换成正解：TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(LinearRegression(), X, y, cv=tscv,
                            scoring='neg_root_mean_squared_error')

print(-scores_ts)
print(-scores_ts.mean())   #rmse
print(scores_ts.std())


'''

回答：

1.这 5 个分数之间的差距，和任务 4 那 5 个比，谁大？大多少？
明显是这个更大，大12

2.前几折和后几折的趋势是什么？（提示：看看它们的顺序）
1-2是降低，2-5就是在升高了

3.那到底哪个"波动"是真实的？任务 4 那个很小的 std，说明了什么？
TimeSeriesSplit 的波动才是真实的。任务4那个极小的 std，是因为k折划分提前看到了未来，所以说数据会很理想








任务 7 · 填表 + 把协议写下来
切分方式	                           RMSE	                      5 折波动（std）
随机切分		                        153                             x
按时间切分		                        209                              x
5 折 KFold(shuffle)	                   156	                           2.06
5 折 TimeSeriesSplit	               168                            34.3
'''


