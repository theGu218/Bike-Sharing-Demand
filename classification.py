import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,precision_score, recall_score, f1_score,roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


df = pd.read_csv('hour.csv')
print(df['cnt'].describe())

'''
先定义【高峰】考虑两种定义：

定义	怎么算	它在问什么
A 按需求绝对量	cnt >= 分位数	"这个小时订车的人多不多"
B 按通勤时段	hr 属于早/晚高峰那几个小时	"现在是不是上下班时间"
从其中选择一个
'''

threshold_75 = df['cnt'].quantile(0.75)
df['A'] = (df['cnt'] >= threshold_75).astype(int)
df['B'] = df['hr'].isin([7, 8, 9, 17, 18, 19]).astype(int)
print('A的正类占比为：',df['A'].mean())
print('B的正类占比为：',df['B'].mean())

both = ((df['A'] == 1)& (df['B'] == 1)).sum()
either = ((df['A'] == 1)| (df['B'] == 1)).sum()
agreement = (df['A'] == df['B']).mean()
print('合集为：',both,'并集为：',either,'一致率为：',agreement)


y = df['B']
hr_dummies = pd.get_dummies(df['hr'], prefix='hr', dtype=float)
print(hr_dummies.head())
print(hr_dummies.shape)
X_c = pd.concat([df[['temp','atemp','hum','windspeed']],hr_dummies], axis=1)
X_train,X_test,y_train,y_test = train_test_split(X_c, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_proba)
print('AUC的值为：',auc)

'''
1.你选哪种定义？为什么？
选择A，因为真实的场景里只关心什么时候用车多而不是在上下班提供

2.两种定义的正类占比几乎一样（都在 25% 上下），那它们能互相替代吗？ 用重合度数字回答。
不能相互替代，因为交集和并集的数字差的太多了,交集2343，并集6378，一致率76.78%

3.任务 3 那个实验的 AUC 是多少？这个数说明了什么？
auc = 1,说明是任务退化，因为输入特征就是hr的独热编码，所以模型根本不需要学习任何规律，只需要看到hr8就直接确定为高峰期


'''
'''

逻辑回归 + 混淆矩阵
用定义 A（cnt >= 你选的阈值）当目标，特征用 P4 那 28 列（4 个数值特征 + hr 独热）。

用同一个主协议 TimeSeriesSplit(n_splits=5)。对每一折：

1.记下 AUC，用 predict_proba 的概率算
2.记下准确率

'''
y = df['A']
result_auc = []

tscv = TimeSeriesSplit(n_splits=5)
for fold,(train_index,test_index) in enumerate(tscv.split(X_c)):
    X_train,X_test = X_c.iloc[train_index], X_c.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_proba) 
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    result_auc.append(auc)   
    print(f"Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}")

print(f"avg: {np.mean(result_auc):.4f}")  
print('===============================')


for C in [0.01,1,100]:
 result_auc1 = []
 for fold,(train_index,test_index) in enumerate(tscv.split(X_c)):
    X_train,X_test = X_c.iloc[train_index], X_c.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C = C ,max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_proba) 
    y_pred = pipe.predict(X_test)
    result_auc1.append(auc)   
 print(f"C = { C } , avg: {np.mean(result_auc1):.4f}") 
'''
回答：

1.5 折的 AUC 分别多少？均值多少？
0.85,0.90,0.90,0.90,0.92,均值是0.8926

2.5 折的差异大吗？和 P3 里回归任务那个 std 比呢？
差异不大，AUC std 0.0224（2.5%）vs RMSE std 34.33（20.3%）

3.5 折的准确率分别多少？它们之间跳得厉害吗？
准确率是0.79,0.86,0.81,0.75,0.74，跳的还挺明显的

4.进阶（可选）：LogisticRegression 默认就带 L2 正则化（参数 C=1.0， C 是 alpha 的倒数）
这是 P5 的知识点换了个马甲。 试试 penalty=None（无正则化）或改几个 C，看 AUC 动不动。
动，但是动的很少


'''

'''

把 5 折的预测概率拼在一起，得到一列完整的概率
按 0.5 切成 0/1
打印混淆矩阵 confusion_matrix(y_true, y_pred)

'''

y = df['A']
y_all_true = []
y_all_proba = []
tscv = TimeSeriesSplit(n_splits=5)
for fold,(train_index,test_index) in enumerate(tscv.split(X_c)):
    X_train,X_test = X_c.iloc[train_index], X_c.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_all_proba.extend(y_proba) 
    y_all_true.extend(y_test)
df_y_proba = np.array(y_all_proba)
df_y_true = np.array(y_all_true)   

df_y_pred = (df_y_proba >= 0.5).astype(int) 
cm = confusion_matrix(df_y_true,df_y_pred)
print('\n===混淆矩阵===')
print(cm)

TN, FP, FN, TP = cm.ravel()
print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")


# precision = 1803 / (1803 + 629) = 0.7414
manual_p = TP / (TP + FP)

# recall = 1803 / (1803 + 2415) = 0.4275
manual_r = TP / (TP + FN)

# f1 = 2 * (P * R) / (P + R)
manual_f1 = 2 * (manual_p * manual_r) / (manual_p + manual_r)

# accuracy = (1803 + 9633) / 14480 = 0.7898
manual_acc = (TP + TN) / (TP + TN + FP + FN)

print(f"\n===手算对拍===")
print(f"Precision: {manual_p:.4f}, Recall: {manual_r:.4f}, F1: {manual_f1:.4f}")
p = precision_score(y_all_true, df_y_pred)
r = recall_score(y_all_true, df_y_pred)
f1 = f1_score(y_all_true,df_y_pred)
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
print(f"accuracy: {manual_acc:.4f} | 全猜不是高峰: {(TN+FP)/len(df_y_true):.4f}")
  
print('===============================')


'''
这个模型 precision 和 recall 谁高谁低，预测更高而回报更低， 说明它偏向还是少报？
把它和"无脑全预测不是高峰"的准确率比一比，高出8%



ROC 曲线 + AUC
用概率画 ROC 曲线，把 AUC 标在图上 → figures/roc_curve.png
图上画一条对角线（AUC=0.5 的基准线，代表"瞎猜"）

TPR = TP/(tp+fn)  Y
FPR = FP/(TN+FP)  X

'''
fpr, tpr, thresholds = roc_curve(df_y_true, df_y_proba)
roc_auc = roc_auc_score(df_y_true, df_y_proba)
plt.figure(1,figsize=(14,7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}(5-fold pooled) ")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('figures/roc_curve.png')

'''
回答：

1.你的 AUC 是多少？怎么用一句话向别人解释这个数？
0.8885,如果从数据里随便抽出两个样本，一个是真实的“高峰”，一个是真实的“平峰”，你的模型给“高峰”打出的概率分数，高于“平峰”的概率分数。这个事件发生的概率，就是 0.8885（88.85%）。

2.曲线靠近左上角说明什么？
越靠近左上角说明模型做的越好，预测能力越强

3.曲线上的每一个点代表什么？ 为什么说 AUC 和阈值无关？
排序固定。阈值只决定切在哪个位置，点是沿同一条曲线移动；曲线的形状由“模型把谁排前面”决定，那个和阈值无关。
每一个点就代表说在特定阈值，对应的fpr and tpr的值

'''

'''
 阈值扫描 + 选一个阈值
默认的 0.5 大概率不是需要的。 扫一遍：

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    
画一张图：横轴阈值、纵轴指标、三条线（P / R / F1），带 legend → figures/threshold_sweep.png

'''
r_p = []
r_r = []
r_f1 = []
t_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in t_array:
    df_y_pred = (df_y_proba >= t).astype(int) 
    cm = confusion_matrix(df_y_true,df_y_pred)
    # print('\n===混淆矩阵===')
    # print('阈值为',t,'\n', cm)

    TN, FP, FN, TP = cm.ravel()
    print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    manual_p = TP / (TP + FP)
    manual_r = TP / (TP + FN)
    manual_f1 = 2 * (manual_p * manual_r) / (manual_p + manual_r)
    manual_acc = (TP + TN) / (TP + TN + FP + FN)
    r_p.append(manual_p)
    r_r.append(manual_r)
    r_f1.append(manual_f1)
    print(f"阈值为：{t}时，Precision: {manual_p:.4f}, Recall: {manual_r:.4f}, F1: {manual_f1:.4f}")

plt.figure(2,figsize = (14,7))
plt.plot(t_array,r_p,label = 'Precision')
plt.plot(t_array,r_r,label = 'Recall')
plt.plot(t_array,r_f1,label = 'F1')
plt.xlabel('threshold')
plt.legend()
plt.savefig('figures/threshold_sweep.png')
plt.show()



'''

回答：

阈值变大时，precision 往哪走？recall 往哪走？ 为什么是这个方向？
p在增大，r在减小,阈值变高则导致fp变低，tp降低，fn增大

F1 在哪个阈值最大？ 比 0.5 时的 F1 好了多少？
0.1的时候最大，为0.72，比0.5的好了（0.72-0.54）/0.54 = 33%

注意 0.9 那一行 —— precision 反而掉下来了。为什么？
是因为预测为“是”的样本基数太小了，只要里面混进几个错的，精确率就会大跌。

业务选择：如果这个模型的用途是"提前调度车辆"
漏报（真高峰没报出来）→ 会怎样？
误报（不是高峰却报了）→ 会怎样？
那阈值该往哪边调？ 说出你的理由。
往低了调，如果真高峰但是报出来，会导致没有车可有
如果是不是高峰但误报，也就浪费一些车辆，比漏报损失低

'''
'''
对照：只用 4 个数值特征
把之前的算AUC和ACC的流程重跑一遍，但特征去掉 hr 独热，只用 4 个数值特征。
'''



X = df[['temp','atemp','hum','windspeed']]
y = df['A']
result_auc = []
tscv = TimeSeriesSplit(n_splits=5)
for fold,(train_index,test_index) in enumerate(tscv.split(X)):
    
    X_train,X_test = X.iloc[train_index], X.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_proba) 
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    result_auc.append(auc)   
    print(f"Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}")

print(f"avg_auc: {np.mean(result_auc):.4f}")  
print('===============================')


for C in [0.01,1,100]:
 result_auc1 = []
 for fold,(train_index,test_index) in enumerate(tscv.split(X)):
    X_train,X_test = X.iloc[train_index], X.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C = C ,max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_proba) 
    y_pred = pipe.predict(X_test)
    result_auc1.append(auc)   
 print(f"C = { C } , avg_auc: {np.mean(result_auc1):.4f}") 

    
y_all_true = []
y_all_proba = []
tscv = TimeSeriesSplit(n_splits=5)
for fold,(train_index,test_index) in enumerate(tscv.split(X)):
    X_train,X_test = X.iloc[train_index], X.iloc[test_index]
    y_train,y_test = y.iloc[train_index], y.iloc[test_index]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_train,y_train)
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_all_proba.extend(y_proba) 
    y_all_true.extend(y_test)

df_y_proba = np.array(y_all_proba)
df_y_true = np.array(y_all_true)   
df_y_pred = (df_y_proba >= 0.5).astype(int) 
cm = confusion_matrix(df_y_true,df_y_pred)
# print('\n===混淆矩阵===')
# print(cm)
TN, FP, FN, TP = cm.ravel()
print(f"TN={TN}, FP={FP}, FN={FN}, TP={TP}")
manual_p = TP / (TP + FP)
manual_r = TP / (TP + FN)
manual_f1 = 2 * (manual_p * manual_r) / (manual_p + manual_r)
manual_acc = (TP + TN) / (TP + TN + FP + FN)
print(f"Precision: {manual_p:.4f}, Recall: {manual_r:.4f}, F1: {manual_f1:.4f}")


'''

回答：
AUC 从多少掉到多少？
0.89降到0.74

准确率掉了多少？ 和"无脑全预测不是高峰"的准确率 75% 比，还高多少？
71%,完全不预测全高峰是70.87%，只高了0.3个百分点

一个准确率只比"什么都不做"高零点几个百分点的模型 —— 如果没有 F1 和 AUC，你看得出来它没用吗？
混淆矩阵，漏了3415个数据，漏的太多了。

所以：为什么在类别不平衡的问题上，准确率会骗人？
因为类别不平衡（平峰占 75%）。模型只要学聪明点，稍微往“少报”偏一点，就能蒙混过关拿到很高的准确率，但这掩盖了它在少数类（高峰）上惨不忍睹的表现。

'''


