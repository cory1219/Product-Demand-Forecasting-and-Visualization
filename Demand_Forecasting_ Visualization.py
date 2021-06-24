import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

os.chdir('/Users/mingyutu/Desktop/BMW') 

df = pd.read_csv("Historical Product Demand.csv", parse_dates=['Date'])

print(df.head())
print(df.dtypes)

# remove na or duplication
df.isnull().sum()
df.dropna(inplace=True)
df.reset_index(drop=True)
df.drop_duplicates(inplace=True)

# cleaning
df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")
df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")
df.Order_Demand = df.Order_Demand.astype('int')

# 4 Product type
df = df[df.Product_Code.isin(df.Product_Code.value_counts()[:4].index)]
p1 = df.Product_Code.value_counts()[:4].index[0]
p2 = df.Product_Code.value_counts()[:4].index[1]
p3 = df.Product_Code.value_counts()[:4].index[2]
p4 = df.Product_Code.value_counts()[:4].index[3]

# visualization
g = sns.FacetGrid(df, col="Product_Code")
g.map(sns.distplot, "Order_Demand", fit=norm)

ax1 = plt.subplot(221)
res = stats.probplot(df[df.Product_Code == p1].Order_Demand, plot=plt)
ax2 = plt.subplot(222)
res = stats.probplot(df[df.Product_Code == p2].Order_Demand, plot=plt)
ax3 = plt.subplot(223)
res = stats.probplot(df[df.Product_Code == p3].Order_Demand, plot=plt)
ax4 = plt.subplot(224)
res = stats.probplot(df[df.Product_Code == p4].Order_Demand, plot=plt)
plt.show()

sns.boxplot(df.Product_Code, df.Order_Demand, showfliers=False) # hide outliers

# time-series
df = df.groupby(['Date', 'Product_Code'])['Order_Demand'].sum().reset_index()
df.set_index('Date', inplace=True)
df1 = df[df.Product_Code == p1].resample('MS').sum()
df2 = df[df.Product_Code == p2].resample('MS').sum()
df3 = df[df.Product_Code == p3].resample('MS').sum()
df4 = df[df.Product_Code == p4].resample('MS').sum()

df1.plot() # total sales volumn for the month
df2.plot() # total sales volumn for the month
df3.plot() # total sales volumn for the month
df4.plot() # total sales volumn for the month

# forecast
import itertools
p = d = q = range(1,3)
pdq = list(itertools.product(p, d, q))
pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
result = pd.DataFrame(columns=["MAE%", 'RMSE%'])

result1 = []
for i in pdq:
    for j in pdqs:
        model = sm.tsa.statespace.SARIMAX(
            df1,
            order=i,
            seasonal_order=j)
        fit_model = model.fit()
        result1.append([i, j, fit_model.mae, fit_model.mse])

model = sm.tsa.statespace.SARIMAX(df1,
                                order=(1, 1, 2),
                                seasonal_order=(2, 1, 2, 12))
result1 = model.fit()
result.loc[0] = [(result1.mae/np.mean(df1)).values.item(), (math.sqrt(result1.mse)/np.mean(df1)).values.item()]


result2 = []
for i in pdq:
    for j in pdqs:
        model = sm.tsa.statespace.SARIMAX(
            df2,
            order=i,
            seasonal_order=j)
        fit_model = model.fit()
        result2.append([i, j, fit_model.mae, fit_model.mse])

model2 = sm.tsa.statespace.SARIMAX(df2,
                                order=(2, 1, 2),
                                seasonal_order=(2, 1, 2, 12))
result2 = model.fit()
result.loc[1] = [(result2.mae/np.mean(df2)).values.item(), (math.sqrt(result2.mse)/np.mean(df2)).values.item()]

result3 = []
for i in pdq:
    for j in pdqs:
        model = sm.tsa.statespace.SARIMAX(
            df3,
            order=i,
            seasonal_order=j)
        fit_model = model.fit()
        result3.append([i, j, fit_model.mae, fit_model.mse])

model = sm.tsa.statespace.SARIMAX(df3,
                                order=(2, 1, 2),
                                seasonal_order=(2, 1, 2, 12))
result3 = model.fit()
result.loc[2] = [(result3.mae/np.mean(df3)).values.item(), (math.sqrt(result3.mse)/np.mean(df3)).values.item()]
  
result4 = []
for i in pdq:
    for j in pdqs:
        model = sm.tsa.statespace.SARIMAX(
            df4,
            order=i,
            seasonal_order=j)
        fit_model = model.fit()
        result4.append([i, j, fit_model.mae, fit_model.mse])

model = sm.tsa.statespace.SARIMAX(df4,
                                order=(2, 1, 1),
                                seasonal_order=(1, 1, 1, 12))
result4 = model.fit()
result.loc[3] = [(result4.mae/np.mean(df4)).values.item(), (math.sqrt(result4.mse)/np.mean(df4)).values.item()]
  

pred = result1.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
#Confidence interval.
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = df1['2013':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Order_Demand')
plt.legend()



