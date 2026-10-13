# 🚲 共享单车租借量预测 
>观前提醒：本项目为在读大学生个人学习使用，代码仅供参考，如有帮助非常荣幸。
---
##  项目简介
本项目是基于 UCI 共享单车数据集的机器学习回归实战。目标是通过天气、时间、季节等特征，预测每小时的单车租借量。

##  技术栈
- **语言**：Python 3.14
- **数据处理**：Pandas, NumPy
- **数据可视化**：Matplotlib, Seaborn
- **机器学习**：Scikit-learn (LinearRegression)

##  项目结构
```text
共享单车预测/
├── hour.csv               # 数据集
├── eda.py                  # Part 1: 数据探索与可视化 (EDA)
├── baseline.py                  # Part 2: 特征工程与基线模型建立
├── figures/               # 存放生成的图表 
└── README.md
```

## 数据

| 项 | 情况 |
|---|---|
| 来源 | UCI Machine Learning Repository, *Bike Sharing* (id=275) |
| 下载 | <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip>（273 KB，**免注册**） |
| 文件 | `hour.csv`（小时级，本项目主用）、`day.csv`（按天）、`Readme.txt`（列说明，**先读**） |
| 许可 | CC BY 4.0 —— 可自由转载，注明出处即可 |
| 特征 | 12 个：`season` `yr` `mnth` `hr` `holiday` `weekday` `workingday` `weathersit` `temp` `atemp` `hum` `windspeed` |
| 目标 | `cnt`（总租车数）。另有 `casual`（散客）、`registered`（注册用户）—— **这是目标的分解，不能当特征** |

将 `hour.csv` 放在项目根目录（与 `eda.py` 和 `baseline.py` 同级）。



