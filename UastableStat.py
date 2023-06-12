import pandas as pd#dataframe
import statsmodels.api as sm
import numpy as np#计算
import matplotlib.pyplot as plt
data = pd.read_csv('HW3.csv',parse_dates=['Date'])
data.set_index('Date', inplace=True) # 将日期作为索引
data = data.iloc[:, :-3]
data_sum = data.iloc[:, :5].sum(axis=1)  # 求前五列的和
new_data = pd.concat([data_sum, data.iloc[:, -2:]], axis=1)  # 将前五列的和和原始数据的后两列合并为新的DataFrame
new_data.columns = ['Port', 'SnP', 'Nas']  # 为列添加标签
#print(new_data)

percentage_change = new_data.pct_change()  # 计算百分比变化
#print(percentage_change)
R = percentage_change.drop(percentage_change.index[0])
# print(R)

results1 = pd.DataFrame(columns=['Date', 'beta'])
beta_listSnP = []
results2 = pd.DataFrame(columns=['Date', 'beta'])
beta_listNas = []
window_size = 90
# 循环计算从第一个时间点开始每个时间点往后90个观测值的beta
for i in range(window_size - 1, len(R)):
    # 提取当前区间的数据
    subset = R.iloc[i - window_size + 1: i]

    # 计算beta
    model1 = sm.OLS.from_formula(formula='Port ~ SnP', data=subset).fit()
    model2 = sm.OLS.from_formula(formula='Port ~ Nas', data=subset).fit()
    B1 = model1.params['SnP']
    B2 = model2.params['Nas']
    beta_listSnP.append(B1)
    beta_listNas.append(B2)
results1['Date'] = R.index[window_size - 1:].values
results1['beta'] = beta_listSnP
results2['Date'] = R.index[window_size - 1:].values
results2['beta'] = beta_listNas
# print(results1)
# print(results2)
# Diagram
import matplotlib.pyplot as plt
# 提取日期和Beta值
dates = results1['Date']
beta_snp = results1['beta']
beta_nas = results2['beta']
# 创建图表
plt.figure(figsize=(10, 6))
# 绘制Beta值折线图
plt.plot(dates, beta_snp, label='SnP Beta')
plt.plot(dates, beta_nas, label='Nas Beta')
# 添加标题和标签
plt.title('Beta Analysis')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.legend()
# 显示图表
plt.show()
# P value
p_value_snp = model1.pvalues['SnP']
p_value_nas = model2.pvalues['Nas']
#print(p_value_snp,p_value_nas)
# R方
r_squared_snp = model1.rsquared
r_squared_nas = model2.rsquared
#print("R-squared for SnP:", r_squared_snp)
#print("R-squared for Nas:", r_squared_nas)

#计算180天波动率
window_size = 180
result_stats = pd.DataFrame(columns=['Date', 'Mean', 'Vol'])
mean_list = []
vol_list = []
# 循环计算每个时间点往后180个观测值的波动率
for i in range(window_size, len(R)): # 从第181个数开始取结果，index是180
    # 提取当前区间的数据
    subsetvol = R['Port'].iloc[i - window_size + 1: i-1]
    # 计算均值
    mean = np.mean(subsetvol)
    mean_list.append(mean)
    # 计算波动率
    vol = np.std(subsetvol)
    vol_list.append(vol)
result_stats['Date'] = R.index[window_size:].values
result_stats['Mean'] = mean_list
result_stats['Vol'] = vol_list
result_stats.set_index('Date', inplace=True) # 将日期作为索引
# print(result_stats)
# VaR
Z = 1.645
VaR =  (- (result_stats['Mean'] - Z * result_stats['Vol']))
print(VaR)

# 绘制波动率随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(result_stats.index, result_stats['Vol'])
plt.title('Volatility over Time')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.xticks(rotation=45)
plt.show()
# 绘制VaR随时间的变化
plt.figure(figsize=(12, 6))
plt.plot(result_stats.index, VaR)
plt.title('VaR over Time')
plt.xlabel('Date')
plt.ylabel('VaR')
plt.xticks(rotation=45)
plt.show()
# 找到最大的VaR
max_var_date = VaR.idxmax()
max_var_value = VaR.max()
print(max_var_date, max_var_value)

# Determine if this model backtests successfully
from scipy.stats import binom
Backtesting = pd.DataFrame()
Backtesting['Port'] = R['Port'][180:]
Backtesting['VaR'] = -VaR.values
Exceedance = (Backtesting['VaR'] > Backtesting['Port']).sum()
n = len(Backtesting)
p = 0.05
confidence = 0.95
k = binom.ppf(confidence, n, p)
print('Exceedance', Exceedance)
print('K', k)
