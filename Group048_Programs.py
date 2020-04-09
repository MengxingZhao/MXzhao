import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
daily = pd.read_csv('ASX200Daily.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
dateparse1 = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
monthly = pd.read_csv('ASX200Monthly.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
daily_close = daily['Close']
monthly_close = monthly['Close']
daily.head()
monthly.head()
#%%
print('ASX 200 daily data information:')
print(daily.info())
print('ASX 200 monthly data information:')
print(monthly.info())
print('ASX 200 daily data description:')
print(daily.describe())
print('ASX 200 monthly data description:')
print(monthly.describe())

#%%
daily.isnull().sum()
monthly.isnull().sum()

daily = daily.interpolate()

monthly = monthly.interpolate()
Daily = daily['Close']
Monthly = monthly['Close']
daily = pd.DataFrame(Daily.dropna())
monthly = pd.DataFrame(Monthly.dropna())
#%%
plt.figure()
plt.plot(daily)
plt.title('daily index of ASX200')
plt.xlabel('Date')
plt.ylabel('Index')
plt.show()
#%%
plt.figure()
plt.plot(monthly)
plt.title('monthly index of ASX200')
plt.xlabel('Date')
plt.ylabel('Index')
plt.show()
#
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
ASX200Daily = pd.read_csv('ASX200Daily.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
ASX200Monthly = pd.read_csv('ASX200Monthly.csv', parse_dates=['Date'], index_col='Date', date_parser=dateparse)
print(ASX200Daily.head())
print(ASX200Monthly.head())
ASX200Daily.isnull().sum()
ASX200Daily = ASX200Daily.interpolate()
ASX200Daily.isnull().sum()
daily = pd.DataFrame(ASX200Daily['Close'])
Monthly = ASX200Monthly['Close']
monthly = pd.DataFrame(Monthly.dropna())
#decompositions
#daily
#1.Initial trend estimate
Trend1 = daily.rolling(2, center = True).mean().rolling(250, center = True).mean()
plt.figure()
plt.plot(daily, color='blue', label='daily')
plt.plot(Trend1, color='red',label='Trend1')
plt.title('daily trend')
plt.legend()
#2.de-trend:multipul
daily_res1 = daily/Trend1
plt.figure()
plt.plot(daily_res1)
plt.title('daily multiple')
#seasonal index
daily_res_1 = daily_res1.iloc[128:,:]
daily_res_zero_mul = np.nan_to_num(daily_res_1)
daily_S_mul = np.reshape(daily_res_zero_mul,(19,250))
daily_avg_mul = np.mean(daily_S_mul, axis=0)
print(daily_avg_mul)
#2.de-trend:add
daily_res2 = daily-Trend1
plt.figure()
plt.plot(daily_res2)
plt.title('daily add')
#seasonal index
daily_res_2 = daily_res2.iloc[128:,:]
daily_res_zero_add = np.nan_to_num(daily_res_2)
daily_S_add = np.reshape(daily_res_zero_add,(19,250))
daily_avg_add = np.mean(daily_S_add, axis=0)
print(daily_avg_add)
#monthly
#1.Initial trend estimate
Trend2 = monthly.rolling(2, center = True).mean().rolling(12, center = True).mean()
plt.figure()
plt.plot(monthly, color='blue', label='monthly')
plt.plot(Trend2, color='red',label='Trend2')
plt.title('monthly trend')
plt.legend()
#2.de-trend:multipul
monthly_res1 = monthly/Trend2
plt.figure()
plt.plot(monthly_res1)
plt.title('monthly multiple')
#seasonal index
monthly_res_1 = monthly_res1.iloc[2:,:]
monthly_res_zero_mul = np.nan_to_num(monthly_res_1)
monthly_S_mul = np.reshape(monthly_res_zero_mul,(19,12))
monthly_avg_mul = np.mean(monthly_S_mul, axis=0)
print(monthly_avg_mul)
#2.de-trend:add
monthly_res2 = monthly-Trend2
plt.figure()
plt.plot(monthly_res2)
plt.title('monthly add')
#seasonal index
monthly_res_2 = monthly_res2.iloc[2:,:]
monthly_res_zero_add = np.nan_to_num(monthly_res_2)
monthly_S_add = np.reshape(monthly_res_zero_add,(19,12))
monthly_avg_add = np.mean(monthly_S_add, axis=0)
print(monthly_avg_add)
# split the data set 
train_size_daily = int(len(daily['Close'])*0.9)
validation_size_daily = len(daily['Close'])-train_size_daily
daily_train = daily.iloc[:train_size_daily,:]
daily_val = daily.iloc[train_size_daily:,:]
train_size_monthly = int(len(monthly['Close'])*0.9)
validation_size_monthly = len(monthly['Close'])-train_size_monthly
monthly_train = monthly.iloc[:train_size_monthly,:]
monthly_val = monthly.iloc[train_size_monthly:,:]
#
#Naive forecast for daily
nai1 = np.asarray(daily_train['Close'])
y_d = daily_val.copy()
y_d['naive'] = nai1[len(nai1)-1]
plt.figure()
plt.plot(daily_train.index, daily_train['Close'], label='train')
plt.plot(daily_val.index, daily_val['Close'], label='validation')
plt.plot(y_d.index, y_d['naive'], label='forecast')
plt.title('Naive Forecast for daily')
plt.legend()
#MSE
from sklearn.metrics import mean_squared_error
print("MSE_daily_naive: {0}".format(mean_squared_error(daily_val['Close'], y_d['naive'])))
#
#Naive forecast for monthly
nai2 = np.asarray(monthly_train['Close'])
y_m = monthly_val.copy()
y_m['naive'] = nai2[len(nai2)-1]
plt.figure()
plt.plot(monthly_train.index, monthly_train['Close'], label='train')
plt.plot(monthly_val.index, monthly_val['Close'], label='validation')
plt.plot(y_m.index, y_m['naive'], label='forecast')
plt.title('Naive Forecast for monthly')
plt.legend()
#MSE
print("MSE_monthly_naive: {0}".format(mean_squared_error(monthly_val['Close'], y_m['naive'])))
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#load the data as a time series
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
daily = pd.read_csv('ASX200Daily.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
monthly = pd.read_csv('ASX200Monthly.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
daily.info()
monthly.info()
daily.describe()
monthly.describe()
print(daily)
print(monthly)
#clean data
orig=pd.read_csv('ASX200Daily.csv')
orig['Date']= pd.to_datetime(orig['Date'])
orig.set_index('Date',inplace=True)
orig.isnull().sum()
orig = orig.interpolate()
orig.isnull().sum()
clean_daily=orig['Close']
index_daily=clean_daily.to_frame()
clean_monthly=monthly.dropna()['Close']
index_monthly=clean_monthly.to_frame()
clean_daily.describe()
clean_monthly.describe()
print(clean_daily)
print(clean_monthly)
#train-validation split
train_size1 = int(len(index_daily['Close'])*0.9)
validation_size1 = len(index_daily['Close'])-train_size1
train_daily=clean_daily.iloc[:train_size1]
validation_daily=clean_daily.iloc[train_size1:]
train_size2 = int(len(index_monthly['Close'])*0.9)
validation_size2 = len(index_monthly['Close'])-train_size2
train_monthly=clean_monthly.iloc[:train_size2]
validation_monthly=clean_monthly.iloc[train_size2:]
#Drift Method
#daily
y_hat_drift_d=validation_daily.copy().to_frame()
y_t_d=train_daily.iloc[-1]
y_1_d=train_daily.iloc[0]
y_hat_drift_d['drift_d']=0
for i in range(len(validation_daily)):
     y_hat_drift_d['drift_d'].iloc[i]=y_t_d+((i+1)*(y_t_d-y_1_d)/(len(train_daily)-1))
     i=i+1   
print(y_hat_drift_d['drift_d'] )
plt.figure(figsize=(12,8))
plt.plot(train_daily, label='train_d')
plt.plot(validation_daily, label='validation_d')
plt.plot(y_hat_drift_d['drift_d'], label='drift_forecast_d')
plt.legend(loc='best')
plt.show()
mse_drift_d=mean_squared_error(validation_daily,y_hat_drift_d['drift_d'])
print(mse_drift_d)
#prediction for next five days
drift_pre_d=[]
y_t_d1=validation_daily.iloc[-1]
y_1_d1=train_daily.iloc[0]
for m in range(5):
     drift_pre_d.append(y_t_d1+((m+1)*(y_t_d1-y_1_d1)/(len(clean_daily)-1)))
print(drift_pre_d)
#monthly
y_hat_drift_m= validation_monthly.copy().to_frame()
y_t_m=train_monthly.iloc[-1]
y_1_m=train_monthly.iloc[0]
y_hat_drift_m['drift_m']=0
for k in range(len(validation_monthly)):
     y_hat_drift_m['drift_m'].iloc[k]=y_t_m+((k+1)*(y_t_m-y_1_m)/(len(train_monthly)-1))
     k=k+1   
print(y_hat_drift_m['drift_m'] )
plt.figure(figsize=(12,8))
plt.plot(train_monthly, label='train_m')
plt.plot(validation_monthly, label='validation_m')
plt.plot(y_hat_drift_m['drift_m'], label='drift_forecast_m')
plt.legend(loc='best')
plt.show()
mse_drift_m=mean_squared_error(validation_monthly,y_hat_drift_m['drift_m'])
print(mse_drift_m)
#prediction for next five months
drift_pre_m=[]
y_t_m1=validation_monthly.iloc[-1]
y_1_m1=train_monthly.iloc[0]
for n in range(5):
     drift_pre_m.append(y_t_m1+((n+1)*(y_t_m1-y_1_m1)/(len(clean_monthly)-1)))
print(drift_pre_m)
#Holt's linear method
#daily
#define function
def holts(y,alpha,beta):
    ll=[y[0]]
    bb=[y[1]-y[0]]
    holts_mannually=[]
    for t in range(len(y)):
        ll.append(alpha*y[t]+(1-alpha)*(ll[t]+bb[t]))
        bb.append(beta*(ll[t+1]-ll[t])+(1-beta)*bb[t])
        holts_mannually.append(ll[t+1]+bb[t+1])
    return holts_mannually,ll[1:],bb[1:]
def sse(x,y):
    return np.sum(np.power(x-y,2))
sse_d=[]
alphas=np.arange(0.1,1,0.1)
betas=np.arange(0.1,1,0.1)
for a in alphas:
    for b in betas:
        smoothed_d,ll_d,bb_d=holts(clean_daily,alpha=a,beta=b)
        sse_d.append(sse(smoothed_d[:-1],clean_daily.values[1:]))      
#optimal alpha and beta
optimal_d = np.argmin(sse_d)
#def com()
com=[]
alphas=np.arange(0.1,1,0.1)
betas=np.arange(0.1,1,0.1)
for p in alphas:
    for q in betas:
        com.append((p,q))
print(com[optimal_d])
#calculte MSE
holts_train_d,lll,bbb=holts(train_daily,alpha=0.9,beta=0.1)
holts_val_d=[]
for day in range(len(validation_daily)):
    holts_val_d.append(lll[-1]+(day+1)*bbb[-1])
print(holts_val_d)
mse_holts_d=mean_squared_error(holts_val_d,validation_daily)
print(mse_holts_d)
#forecast
holts_train_dd,l,b=holts(clean_daily,alpha=0.9,beta=0.1)
holts_fore_d=[]
for da in range(5):
    holts_fore_d.append(l[-1]+(da+1)*b[-1])
print(holts_fore_d)
#monthly
#define function
sse_m=[]
alphas=np.arange(0.1,1,0.1)
betas=np.arange(0.1,1,0.1)
for a in alphas:
    for b in betas:
        smoothed_m,ll_m,bb_m=holts(clean_monthly,alpha=a,beta=b)
        sse_m.append(sse(smoothed_m[:-1],clean_monthly.values[1:]))
#optimal alpha and beta
optimal_m = np.argmin(sse_m)
print(com[optimal_m])
#calculte MSE
holts_train_m,lll,bbb=holts(train_monthly,alpha=0.9,beta=0.1)
holts_val_m=[]
for month in range(len(validation_monthly)):
    holts_val_m.append(lll[-1]+(month+1)*bbb[-1])
print(holts_val_m)
mse_holts_d=mean_squared_error(holts_val_m,validation_monthly)
print(mse_holts_m)
#forecast
holts_train_mm,l,b=holts(clean_monthly,alpha=0.9,beta=0.1)
holts_fore_m=[]
for mon in range(5):
    holts_fore_m.append(l[-1]+(mon+1)*b[-1])
print(holts_fore_m)
#%%------------------------------------------------------------------------
#                               ARIMA
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot
import warnings
warnings.filterwarnings('ignore')

#------------------daily data and IDA-------------------------

orig=pd.read_csv('ASX200Daily.csv')
orig['Date']= pd.to_datetime(orig['Date'])
orig.set_index('Date',inplace=True)

# clean the data
#filling missing value
orig.isnull().sum()
orig = orig.interpolate()
orig.isnull().sum()


# split the data set 
train_size = int(len(orig['Close'])*0.9)
validation_size = len(orig['Close'])-train_size

train_d = orig.iloc[:train_size,:]
validation_d = orig.iloc[train_size:,:]


#%%
#----------------------data prepocessing----------------------

#check initial training data
#plot original data
D=train_d.index
p=train_d['Close']
plt.figure(figsize=(8,4))
lines=plt.plot(D,p)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The original daily data')
plt.xlabel('Datetime')
plt.ylabel('Close price')

#log transform
train_d['close_log']=np.log(train_d['Close'])
D=train_d.index
p=train_d['close_log']
plt.figure(figsize=(8,4))
lines=plt.plot(D,p)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The Log transformed daily data')
plt.xlabel('Datetime')
plt.ylabel('Log close price')

#first differencing
close_log=train_d['close_log']
close_log_diff=train_d['close_log'].diff()
close_log_diff.dropna(inplace=True)
D=train_d.index[1:]
x=close_log_diff
plt.figure(figsize=(8,5))
lines=plt.plot(D,x)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The first differece of log-transformed data')
plt.xlabel('Datetime')
plt.ylabel('First difference')
plt.show()

#test stationarity after log and first difference
#plot ACF & PACF
plt.figure(figsize=(20,8))
acf=sm.graphics.tsa.plot_acf(close_log_diff,lags=30, alpha = 0.05)
pacf=sm.graphics.tsa.plot_pacf(close_log_diff,lags=30, alpha = 0.05)

#Dickey-Fuller test
def test_stationarity(timeseries):
    print('Results of Dickey-Fuller test')
    result=adfuller(timeseries,autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Lages Used: %f' % result[2])
    print('Number of Observations Used: %f' % result[3])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
test_stationarity(close_log_diff)
#stationary, so tentative order (3,i,3)=(3,1,3)

#%% 
#----------------------order selection---------------------- 
#Select order by AIC
import statsmodels.tsa.stattools as st
order =st.arma_order_select_ic(close_log_diff,max_ar=6,max_ma=6,ic=['aic'])
print(order.aic_min_order)
p=order.aic_min_order[0]
q=order.aic_min_order[1]


#%%
#------------------------model fitting----------------------------
# ARIMA(6,1,6) model
from statsmodels.tsa.arima_model import ARIMA
model1 = ARIMA(close_log, order=(4,1,3))
result1 = model1.fit(disp=-1)
print(result1.summary())

# ARIMA(3,1,3) model
from statsmodels.tsa.arima_model import ARIMA
model2 = ARIMA(close_log, order=(3, 1, 3))
result2 = model2.fit(disp=-1)
print(result2.summary())
#%%
#------------------------residual diagonostic-----------------
# plot residuals
residuals_d = pd.DataFrame(result1.resid)
plt.figure(figsize=(16,4))
plt.plot(residuals_d)
plt.title('Residuals')
plt.show()

#plot ACF for residual
plt.figure()
acf_r=sm.graphics.tsa.plot_acf(residuals_d,lags=30, alpha = 0.05)

# plot distribution of residual
sns.distplot(residuals_d)

#test the stationary of residuals
#Dickey-Fuller test
test_stationarity(residuals_d.loc[:,0])

#test the normality of residuals
#KS test
from scipy.stats import kstest
x=kstest(residuals_d, 'norm')

#the overall description of residual
residuals_d.describe()

#%%
#-------------invertability and stationarity test for MA------
ma = np.array([3])
arma_process = sm.tsa.ArmaProcess(ma)
print("MA Model is{0}stationary".format(" "if arma_process.isstationary else "not"))
print("MA Model is{0}invertible".format(" "if arma_process.isinvertible else "not"))

#%%
#-----------------get fitted series and training MSE---------------
# Get Fitted Series in initial form
forecast_t = result1.predict(typ = 'levels', dynamic =False)
train_d['forecast']=None
train_d.iloc[1:,7]=np.exp(forecast_t)
train_d.head()

# plot fitted series
D=train_d.index
p=train_d['Close']
f=train_d['forecast']
plt.figure(figsize=(8,4))
lines=plt.plot(D,f,p)
plt.setp(lines[0],linewidth=2,color='red',linestyle='--')
plt.setp(lines[1],linewidth=0.7,color='blue')
plt.title('The fitted daily series')
plt.xlabel('Datetime')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()

# caculate MSE
from sklearn.metrics import mean_squared_error
train_d_mse=mean_squared_error(train_d.iloc[1:,7],train_d.iloc[1:,3])

print('ARIMA train daily data MSE:{0:2f}'.format(train_d_mse))


#%%
#-------------- forecast for validation data set------------------
validation_d['forecast']=None
validation_d['forecast']=np.exp(result1.forecast(steps=validation_size)[0])

#visulazition
D=validation_d.index
p=validation_d['Close']
f=validation_d['forecast']
plt.figure(figsize=(8,4))
lines=plt.plot(D,f,p)
plt.setp(lines[0],linewidth=1,color='red')
plt.setp(lines[1],linewidth=1,color='blue')
plt.title('The forecast for validation set')
plt.xlabel('Datetime')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()

# Validation MSE 
validation_d_mse=mean_squared_error(validation_d['forecast'],validation_d['Close'])
print('ARIMA validation daily data MSE:{0:2f}'.format(validation_d_mse))
#%%
#------------------ 5 day forecast--------------------- 
orig['close_log']=np.log(orig['Close'])
from statsmodels.tsa.arima_model import ARIMA
model3 = ARIMA(orig['close_log'], order=(4,1,3))
result3 = model3.fit(disp=-1)
print(result3.summary())

# save as csv
x=np.around(np.exp(result3.forecast(steps=5)[0]), decimals=2)
dt=pd.date_range(start='2019-05-27', end='2019-05-31')
pre=pd.DataFrame({'Predictions':x,'Dates':dt})
pre.set_index('Dates',inplace=True)
pre.to_csv('prediction_ARIMA.csv')


#%% ------------------------monthly data------------------------------------
orig_m=pd.read_csv('ASX200Monthly.csv')
orig_m['Date']= pd.to_datetime(orig_m['Date'])
orig_m.set_index('Date',inplace=True)

# clean the data
#filling missing value
orig_m.isnull().sum()
orig_m= orig_m.interpolate()
orig_m.isnull().sum()


# split the data set 
train_size_m = int(len(orig_m['Close'])*0.9)
validation_size_m = len(orig_m['Close'])-train_size_m

train_m = orig_m.iloc[:train_size_m,:]
validation_m = orig_m.iloc[train_size_m:,:]


#%%
#----------------------data prepocessing----------------------

#check initial training data
#plot original data
D=train_m.index
p=train_m['Close']
plt.figure(figsize=(8,4))
lines=plt.plot(D,p)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The original monthly data')
plt.xlabel('Datetime')
plt.ylabel('Close price')

#log transform
train_m['close_log']=np.log(train_m['Close'])
D=train_m.index
p=train_m['close_log']
plt.figure(figsize=(8,4))
lines=plt.plot(D,p)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The Log transformed monthly data')
plt.xlabel('Datetime')
plt.ylabel('Log close price')

#first differencing
close_log_m=train_m['close_log']
close_log_diff_m=train_m['close_log'].diff()
close_log_diff_m.dropna(inplace=True)
D=train_m.index[1:]
x=close_log_diff_m
plt.figure(figsize=(8,5))
lines=plt.plot(D,x)
plt.setp(lines[0],linewidth=1,color='green')
plt.title('The first differece of log-transformed data')
plt.xlabel('Datetime')
plt.ylabel('First difference')
plt.show()

#test stationarity after log and first difference
#plot ACF & PACF
plt.figure(figsize=(20,8))
acf=sm.graphics.tsa.plot_acf(close_log_diff_m,lags=30, alpha = 0.05)
pacf=sm.graphics.tsa.plot_pacf(close_log_diff_m,lags=30, alpha = 0.05)

#Dickey-Fuller test
def test_stationarity(timeseries):
    print('Results of Dickey-Fuller test')
    result=adfuller(timeseries,autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Lages Used: %f' % result[2])
    print('Number of Observations Used: %f' % result[3])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
test_stationarity(close_log_diff_m)
#stationary, so tentative order (3,i,3)=(3,1,3)

#%% 
#----------------------order selection---------------------- 
#Select order by AIC
import statsmodels.tsa.stattools as st
order =st.arma_order_select_ic(close_log_diff_m,max_ar=6,max_ma=6,ic=['aic'])
print(order.aic_min_order)
p=order.aic_min_order[0]
q=order.aic_min_order[1]


#%%
#------------------------model fitting----------------------------
# ARIMA(6,1,6) model
from statsmodels.tsa.arima_model import ARIMA
model1_m = ARIMA(close_log_m, order=(5,1,2))
result1_m = model1_m.fit(disp=-1)
print(result1_m.summary())
#%%
# ARIMA(3,1,3) model
from statsmodels.tsa.arima_model import ARIMA
model2_m = ARIMA(close_log_m, order=(0, 1, 0))
result2_m = model2_m.fit(disp=-1)
print(result2_m.summary())
#%%
#------------------------residual diagonostic-----------------
# plot residuals
residuals_m = pd.DataFrame(result1_m.resid)
plt.figure(figsize=(16,4))
plt.plot(residuals_m)
plt.title('Residuals')
plt.show()

#plot ACF for residual
plt.figure()
acf_r=sm.graphics.tsa.plot_acf(residuals_m,lags=30, alpha = 0.05)

# plot distribution of residual
sns.distplot(residuals_m)

#test the stationary of residuals
#Dickey-Fuller test
test_stationarity(residuals_m.loc[:,0])

#test the normality of residuals
#KS test
from scipy.stats import kstest
x=kstest(residuals_m, 'norm')

#the overall description of residual
residuals_m.describe()

#%%
#-------------invertability and stationarity test for MA------
ma = np.array([2])
arma_process = sm.tsa.ArmaProcess(ma)
print("MA Model is{0}stationary".format(" "if arma_process.isstationary else "not"))
print("MA Model is{0}invertible".format(" "if arma_process.isinvertible else "not"))

#%%
#-----------------get fitted series and training MSE---------------
# Get Fitted Series in initial form
forecast_t_m = result1_m.predict(typ = 'levels', dynamic =False)
train_m['forecast']=None
train_m.iloc[1:,7]=np.exp(forecast_t_m)
train_m.head()

# plot fitted series
D=train_m.index
p=train_m['Close']
f=train_m['forecast']
plt.figure(figsize=(8,4))
lines=plt.plot(D,f,p)
plt.setp(lines[0],linewidth=1,color='red',label='forecast price')
plt.setp(lines[1],linewidth=0.7,color='blue',label='actual price')
plt.title('The fitted monthly series')
plt.xlabel('Datetime')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()

# caculate RMSE
train_m_mse=mean_squared_error(train_m.iloc[1:,7],train_m.iloc[1:,3])

print('ARIMA train daily data RMSE:{0:2f}'.format(train_m_mse))


#%%
#-------------- forecast for validation data set------------------
validation_m['forecast']=None
validation_m['forecast']=np.exp(result1_m.forecast(steps=validation_size_m)[0])

#visulazition
D=validation_m.index
p=validation_m['Close']
f=validation_m['forecast']
plt.figure(figsize=(8,4))
lines=plt.plot(D,f,p)
plt.setp(lines[0],linewidth=1,color='red',label='forecast price')
plt.setp(lines[1],linewidth=1,color='blue',label='actual price')
plt.title('The forecast for validation set')
plt.xlabel('Datetime')
plt.ylabel('price')
plt.legend(loc='upper left')
plt.show()

# Validation RMSE 
validation_m_mse=mean_squared_error(validation_m['forecast'],validation_m['Close'])
print('ARIMA validation daily data MSE:{0:2f}'.format(validation_m_mse))
#%%
#------------------ 5 day forecast--------------------- 
orig_m['close_log']=np.log(orig_m['Close'])
from statsmodels.tsa.arima_model import ARIMA
model3_m = ARIMA(orig_m['close_log'], order=(5,1,2))
result3_m = model3_m.fit(disp=-1)
print(result3_m.summary())
print(np.exp(result3_m.forecast(steps=5)[0]))
#NN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers.core import Dense 
from keras.models import Sequential
#%%
# ignore warning
import warnings
warnings.simplefilter('ignore')
# set figsize
from pylab import rcParams
rcParams['figure.figsize'] = 16,6
np.random.seed(1)
data_d = pd.read_csv('ASX200Daily.csv', parse_dates=['Date'],index_col='Date')
data_m = pd.read_csv('ASX200Monthly.csv', parse_dates=['Date'],index_col='Date')
data_d_cls= data_d['Close']
data_m_cls= data_m['Close']
data_d_cls=pd.to_numeric(data_d_cls,errors='coerce')
data_m_cls=pd.to_numeric(data_m_cls,errors='coerce')
data_d_cls=data_d_cls.interpolate()
data_m_cls=data_m_cls.dropna()
data_m_cls=data_m_cls.values.reshape((len(data_m_cls),1))
data_d_cls=data_d_cls.values.reshape((len(data_d_cls),1))
scaler = MinMaxScaler(feature_range=(0, 1))
#%%
data_m_cls = scaler.fit_transform(data_m_cls)
data_d_cls = scaler.fit_transform(data_d_cls)
# define window size = 6, daily
time_window_d = 6
#%%

Xall_d, Yall_d = [], []
for i in range(time_window_d, len(data_d_cls)):
    Xall_d.append(data_d_cls[i-time_window_d:i, 0])
    Yall_d.append(data_d_cls[i, 0])

Xall_d = np.array(Xall_d)    
Yall_d = np.array(Yall_d)
train_size = int(len(Xall_d) * 0.9)
test_size = len(Xall_d) - train_size

# split training set
Xtrain = Xall_d[:train_size,:] #
Ytrain = Yall_d[:train_size] 
Xtest = Xall_d[-test_size:,:]  # 
Ytest = Yall_d[-test_size:]     # 
#%%
####   grid search start
best_score = 100000
for a in [60]:
    for b in [30]:
        for c in [12,14]:
            model = Sequential()
            model.add(Dense(c, input_dim = time_window_d, activation = 'relu'))
            model.add(Dense(1))
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')
            np.random.seed(1)
            model.fit(Xtrain, Ytrain, epochs = b, batch_size = a, verbose = 2, validation_split = 0.05) # use 95% for training and 5% for testing
            # predict all y values and transform back to normal scale
            allPredict = model.predict(Xall_d)
            allPredictPlot = scaler.inverse_transform(allPredict)
            trainScore = mean_squared_error(scaler.inverse_transform(data_d_cls[:train_size]), allPredictPlot[:train_size,0])
            print('Training Data MSE: {0:.2f}'.format(trainScore))
            if trainScore < best_score:
                best_score = trainScore
                best_parameters = {'epochs':b,'batch_size':a,'neauon':c}
            else :
                best_score =best_score
            
print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
        
#%%
 
#%%
# add two layers (1 hidden + 1 output)
model_d = Sequential()
model_d.add(Dense(14, input_dim = time_window_d, activation = 'relu'))
model_d.add(Dense(1))
# loss function: MSE
# optimization method: ADAM
model_d.compile(loss = 'mean_squared_error', optimizer = 'adam')
np.random.seed(1) 


#%%
model_d.fit(Xtrain, Ytrain, 
          epochs = 30, 
          batch_size = 60, 
          verbose = 2, 
          validation_split = 0.05) # use 95% for training and 5% for testing
#%%
allPredict = model_d.predict(Xall_d)
allPredictPlot = scaler.inverse_transform(allPredict)

#%%
# visualization

#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls), label='True Data')
plt.plot(np.arange(time_window_d, len(data_d_cls)), allPredictPlot, label='One-Step Prediction') 
plt.title('Feedforward Neural Network daily data in One step forecast')
plt.legend();
plt.xlabel('Days')
plt.ylabel('price')


#%%
trainScore = mean_squared_error(Ytrain, allPredict[:train_size,0])
print('Training Data MSE: {0:.8f}'.format(trainScore))
#%%
#dynamic forcast
dynamic_prediction = np.copy(data_d_cls[:len(data_d_cls) - test_size]) 

for i in range(len(data_d_cls) - test_size, len(data_d_cls)):
    last_feature = np.reshape(dynamic_prediction[i-time_window_d:i], (1,time_window_d))
    next_pred = model_d.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)


#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls[:len(data_d_cls) - test_size]), label='Training Data')
plt.plot(np.arange(len(data_d_cls) - test_size, len(data_d_cls), 1), 
         scaler.inverse_transform(data_d_cls[-test_size:]), 
         label='Testing Data')
plt.plot(np.arange(len(data_d_cls) - test_size, len(data_d_cls), 1), 
         dynamic_prediction[-test_size:], 
         label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.xlabel('Days')
plt.ylabel('price')
plt.title('Dynamic Forecasting results (Feedforward NN in daily data)');

#%%
testScore = mean_squared_error(scaler.inverse_transform(data_d_cls[-test_size:]), 
                                         dynamic_prediction[-test_size:])
print('Dynamic Forecast MSE: {0:.2f}'.format(testScore))


#%%
trainScore = mean_squared_error(scaler.inverse_transform(data_d_cls[:train_size]), allPredictPlot[:train_size,0])
print('Training Data MSE: {0:.2f}'.format(trainScore))

#%%
origin = np.copy(data_d_cls)
prediction_d = origin

#%%
for i in range(len(prediction_d)-5,len(prediction_d)):
    if  i<=4883:
          last_feature_d= np.reshape(prediction_d[-time_window_d:len(prediction_d)], (1,time_window_d))
          next_pred_d= model_d.predict(last_feature_d)
          prediction_d = np.append(prediction_d, next_pred_d)
    else:
        print(prediction_d)
prediction_d = prediction_d.reshape(-1,1)
prediction_d = scaler.inverse_transform(prediction_d)
#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls), label='Training Data')
plt.plot(np.arange(len(prediction_d) - 5, len(prediction_d), 1), prediction_d[-5:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.title('5 day Forecasting results in dayily data(Feedforward NN)');
plt.xlabel('Days')
plt.ylabel('price')
#%%
print(prediction_d[-5:].round(2))

#%%

#lstn

from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
#%%
# define time window
time_window_d2= 6

# reset training / test data for X and Y
Xall, Yall = [], []

for i in range(time_window_d2, len(data_d_cls)):
    Xall.append(data_d_cls[i-time_window_d2:i, 0])
    Yall.append(data_d_cls[i, 0])

# Convert them from list to array 
Xall = np.array(Xall)      
Yall = np.array(Yall)

train_size = int(len(Xall) * 0.9)
test_size = len(Xall) - train_size

Xtrain = Xall[:train_size, :]
Ytrain = Yall[:train_size]

Xtest = Xall[-test_size:, :]
Ytest = Yall[-test_size:]

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window_d2, 1))  
Xtest = np.reshape(Xtest, (Xtest.shape[0], time_window_d2, 1))  


#%%
from keras.callbacks import EarlyStopping
#%%
####   grid search start
best_score = 100000
for d in [ 60, 80, 100,120,140]:
    for e in [10,15, 20,30,40, 50, 60]:
        for f in [13,14,15,20]:
            model = Sequential()
            model.add(LSTM(input_shape=(None, 1),units=f,return_sequences=False))   
            model.add(Dense(output_dim=1))
            model.add(Activation("linear"))
            model.compile(loss = "mse", optimizer = "rmsprop")
            early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1)
            np.random.seed(1)
         # model fitting
            model.fit(Xtrain,Ytrain,batch_size = d,nb_epoch = e,validation_split = 0.05)
            allPredict = model.predict(np.reshape(Xall, (len(Xall),time_window_d2,1))
            )
            allPredict = scaler.inverse_transform(allPredict)
            allPredictPlot = np.empty_like(data_d_cls)
            allPredictPlot[:, :] = np.nan
            allPredictPlot[time_window_d2:, :] = allPredict
            trainScore = mean_squared_error(scaler.inverse_transform(data_d_cls[:train_size]), allPredict[:train_size,0])
            print('Training Data MSE: {0:.2f}'.format(trainScore))
            if trainScore < best_score:
                best_score = trainScore
                best_parameters = {'epochs':e,'batch_size':d,'unit':f}
            else :
                best_score =best_score

print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
#%%
model = Sequential()


model.add(LSTM(input_shape=(None, 1),units=15,return_sequences=False))   
model.add(Dense(output_dim=1))
model.add(Activation("linear"))

model.compile(loss = "mse", optimizer = "rmsprop")
#%%
early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1)
#%%
np.random.seed(1)

# model fitting
model.fit(Xtrain,
          Ytrain,
          batch_size = 140,
          nb_epoch = 15,
          validation_split = 0.05)
#%%
# predict all y values and transform back to the original scale
allPredict = model.predict(np.reshape(Xall, (len(Xall),time_window_d2,1)))
allPredict = scaler.inverse_transform(allPredict)
# prepare variables for visualization
allPredictPlot = np.empty_like(data_d_cls)
allPredictPlot[:, :] = np.nan
allPredictPlot[time_window_d2:, :] = allPredict
#%%
# visualization results
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls), label='True Data')
plt.plot(allPredictPlot, label='One-Step Prediction') 
plt.legend()
plt.title('LSTM daily data in One step forecast')
plt.xlabel('Days')
plt.ylabel('price')
plt.show()
#%%# MSE score
trainScore = mean_squared_error(scaler.inverse_transform(data_d_cls[:train_size]), 
                                          allPredict[:train_size,0])
print('Training Data MSE: {0:.2f}'.format(trainScore))

#%%
#dynamic forcast
dynamic_prediction = np.copy(data_d_cls[:len(data_d_cls) - test_size])

for i in range(len(data_d_cls) - test_size, len(data_d_cls)):
    last_feature = np.reshape(dynamic_prediction[i-time_window_d2:i], (1,time_window_d2,1))
    next_pred = model.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

# dynamic prediction results, and transform back to the original scale
dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)
#%%
# visualization results
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls[:len(data_d_cls) - test_size]), label='Training Data')
plt.plot(np.arange(len(data_d_cls) - test_size, len(data_d_cls), 1), 
         scaler.inverse_transform(data_d_cls[-test_size:]), 
         label='Testing Data')
plt.plot(np.arange(len(data_d_cls) - test_size, len(data_d_cls), 1), 
         dynamic_prediction[-test_size:], 
         label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.title('LSTM daily data in dynamic forecast')
plt.xlabel('Days')
plt.ylabel('price')
plt.show();
#%%
# MSE score
testScore = mean_squared_error(scaler.inverse_transform(data_d_cls[-test_size:]), 
                                         dynamic_prediction[-test_size:])
print('Dynamic Forecast MSE: {0:.2f}'.format(testScore))

#%%
origin = np.copy(data_d_cls)
prediction_d2 = origin

#%%
for i in range(len(prediction_d2)-5,len(prediction_d2)):
    if  i<=4883:
          last_feature_d2= np.reshape(prediction_d2[-time_window_d2:len(prediction_d2)], (1,time_window_d2,1))
          next_pred_d2= model.predict(last_feature_d2)
          prediction_d2 = np.append(prediction_d2, next_pred_d2)
    else:
        print(prediction_d2)
prediction_d2 = prediction_d2.reshape(-1,1)
prediction_d2 = scaler.inverse_transform(prediction_d2)
#%%
print(prediction_d2[-5:].round(2))

#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_d_cls), label='Training Data')
plt.plot(np.arange(len(prediction_d2) - 5, len(prediction_d2), 1), prediction_d2[-5:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.title('5 day Forecasting results (lstm)')
plt.xlabel('Days')
plt.ylabel('price');
#%%
#month data nn mothod
time_window_m = 12
Xall_m, Yall_m = [], []
for i in range(time_window_m, len(data_m_cls)):
    Xall_m.append(data_m_cls[i-time_window_m:i, 0])
    Yall_m.append(data_m_cls[i, 0])

Xall_m = np.array(Xall_m)    
Yall_m = np.array(Yall_m)
train_size = int(len(Xall_m) * 0.9)
test_size = len(Xall_m) - train_size

# split training set
Xtrain = Xall_m[:train_size,:] #
Ytrain = Yall_m[:train_size] 
Xtest = Xall_m[-test_size:,:]  # 
Ytest = Yall_m[-test_size:]     #
#%%
#grid search start
best_score = 100000
for g in [10, 20, 40, 60, 80, 100]:
    for h in [10, 20,30,50, 60,70,100]:
        for i in [12,14,16,18,20]:
            model_m= Sequential()
            model_m.add(Dense(i, input_dim = time_window_m, activation = 'relu'))
            model_m.add(Dense(1))
            model_m.compile(loss = 'mean_squared_error', optimizer = 'adam')
            np.random.seed(1)
            model_m.fit(Xtrain, Ytrain, epochs = h, batch_size = g, verbose = 2, validation_split = 0.05) # use 95% for training and 5% for testing
            # predict all y values and transform back to normal scale
            allPredict = model_m.predict(Xall_m)
            allPredictPlot = scaler.inverse_transform(allPredict)
            trainScore = mean_squared_error(scaler.inverse_transform(data_m_cls[:train_size]), allPredictPlot[:train_size,0])
            print('Training Data MSE: {0:.2f}'.format(trainScore))
            if trainScore < best_score:
                best_score = trainScore
                best_parameters = {'epochs':h,'batch_size':g,'neauon':i}
            else :
                best_score =best_score
            
print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))

#%%
#%%
# add two layers (1 hidden + 1 output)
model_m1 = Sequential()
model_m1.add(Dense(20, input_dim = time_window_m, activation = 'relu'))
model_m1.add(Dense(1))
# loss function: MSE
# optimization method: ADAM
model_m1.compile(loss = 'mean_squared_error', optimizer = 'adam')
np.random.seed(1)

#%%
model_m1.fit(Xtrain, Ytrain, 
          epochs = 50, 
          batch_size = 100, 
          verbose = 2, 
          validation_split = 0.05) # use 95% for training and 5% for testing
#%%
# predict all y values and transform back to normal scale
allPredict = model_m1.predict(Xall_m)
allPredictPlot = scaler.inverse_transform(allPredict)

#%%
# visualization
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls), label='True Data')
plt.plot(np.arange(time_window_m, len(data_m_cls)), allPredictPlot, label='One-Step Prediction') 
plt.title('Feedforward Neural Network_monly data')
plt.xlabel('months')
plt.ylabel('price')
plt.legend();
#%%
# MSE
trainScore = mean_squared_error(Ytrain, allPredict[:train_size,0])
print('Training Data MSE: {0:.8f}'.format(trainScore))
#%%
#dynamic forcast
dynamic_prediction = np.copy(data_m_cls[:len(data_m_cls) - test_size]) 

for i in range(len(data_m_cls) - test_size, len(data_m_cls)):
    last_feature = np.reshape(dynamic_prediction[i-time_window_m:i], (1,time_window_m))
    next_pred = model_m1.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)


#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls[:len(data_m_cls) - test_size]), label='Training Data')
plt.plot(np.arange(len(data_m_cls) - test_size, len(data_m_cls), 1), 
         scaler.inverse_transform(data_m_cls[-test_size:]), 
         label='Testing Data')
plt.plot(np.arange(len(data_m_cls) - test_size, len(data_m_cls), 1), 
         dynamic_prediction[-test_size:], 
         label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.xlabel('Months')
plt.ylabel('price')
plt.title('Dynamic Forecasting results (Feedforward NN-monthly)');

#%%
testScore = mean_squared_error(scaler.inverse_transform(data_m_cls[-test_size:]), 
                                         dynamic_prediction[-test_size:])
print('Dynamic Forecast MSE: {0:.2f}'.format(testScore))


#%%
trainScore = mean_squared_error(scaler.inverse_transform(data_m_cls[:train_size]), allPredictPlot[:train_size,0])
print('Training Data MSE: {0:.2f}'.format(trainScore))

#%%
origin = np.copy(data_m_cls)
prediction_m = origin

#%%
for i in range(len(prediction_m)-5,len(prediction_m)):
    if  i<=236:
          last_feature_m= np.reshape(prediction_m[-time_window_m:len(prediction_m)], (1,time_window_m))
          next_pred_m= model_m1.predict(last_feature_m)
          prediction_m = np.append(prediction_m, next_pred_m)
    else:
        print(prediction_m)
prediction_m = prediction_m.reshape(-1,1)
prediction_m = scaler.inverse_transform(prediction_m)
#%%
print(prediction_m[-5:].round(2))
#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls), label='Training Data')
plt.plot(np.arange(len(prediction_m) - 5, len(prediction_m), 1), prediction_m[-5:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.xlabel('Months')
plt.ylabel('price')
plt.title('5 month Forecasting results (Feedforward NN)');

#%%
#lstn monthly data

from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
#%%
# define time window
time_window_m2= 12

# reset training / test data for X and Y
Xall_m, Yall_m = [], []

for i in range(time_window_m2, len(data_m_cls)):
    Xall_m.append(data_m_cls[i-time_window_m2:i, 0])
    Yall_m.append(data_m_cls[i, 0])

# Convert them from list to array 
Xall_m = np.array(Xall_m)      
Yall_m = np.array(Yall_m)

train_size = int(len(Xall_m) * 0.9)
test_size = len(Xall_m) - train_size

Xtrain = Xall_m[:train_size, :]
Ytrain = Yall_m[:train_size]

Xtest = Xall_m[-test_size:, :]
Ytest = Yall_m[-test_size:]

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window_m2, 1))  
Xtest = np.reshape(Xtest, (Xtest.shape[0], time_window_m2, 1))  


#%%
from keras.callbacks import EarlyStopping
#%%
####   grid search start
best_score = 100000
for x in [ 60, 80, 100,120,140]:
    for y in [10,15, 20,30,40, 50, 60]:
        for z in [15,20,30,40,50,60,70]:
            model_m2 = Sequential()
            model_m2.add(LSTM(input_shape=(None, 1),units=z,return_sequences=False)) 
            model_m2.add(Dense(output_dim=1))
            model_m2.add(Activation("linear"))
            model_m2.compile(loss = "mse", optimizer = "rmsprop")
            early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1)
            np.random.seed(1)
            model_m2.fit(Xtrain,Ytrain,batch_size = x,nb_epoch = y,validation_split = 0.01)
            allPredict = model_m2.predict(np.reshape(Xall_m, (len(Xall_m),time_window_m2,1))
            )
            allPredict = scaler.inverse_transform(allPredict)
            allPredictPlot = np.empty_like(data_m_cls)
            allPredictPlot[:, :] = np.nan
            allPredictPlot[time_window_m2:, :] = allPredict
            trainScore = mean_squared_error(scaler.inverse_transform(data_m_cls[:train_size]), allPredict[:train_size,0])
            print('Training Data MSE: {0:.2f}'.format(trainScore))
            if trainScore < best_score:
                best_score = trainScore
                best_parameters = {'epochs':y,'batch_size':x,'unit':z}
            else :
                best_score =best_score
print("Best score:{:.2f}".format(best_score))
print("Best parameters:{}".format(best_parameters))
#%%
model_m3= Sequential()
model_m3.add(LSTM(input_shape=(None, 1),units=40,return_sequences=False))
model_m3.add(Dense(output_dim=1))
model_m3.add(Activation("linear"))

model_m3.compile(loss = "mse", optimizer = "rmsprop")
#%%
early_stop = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1)
#%%
np.random.seed(1)
# model fitting
model_m3.fit(Xtrain,
          Ytrain,
          batch_size = 60,
          nb_epoch = 10,
          validation_split = 0.01)
#%%
allPredict = model_m3.predict(np.reshape(Xall_m, (len(Xall_m),time_window_m2,1)))
allPredict = scaler.inverse_transform(allPredict)
allPredictPlot = np.empty_like(data_m_cls)
allPredictPlot[:, :] = np.nan
allPredictPlot[time_window_m2:, :] = allPredict
#%%
# visualization results
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls), label='True Data')
plt.plot(allPredictPlot, label='One-Step Prediction') 
plt.legend()
plt.show()
#%%# MSE score
trainScore = mean_squared_error(scaler.inverse_transform(data_m_cls[:train_size]), 
                                          allPredict[:train_size,0])
print('Training Data MSE: {0:.2f}'.format(trainScore))

#%%
#dynamic forcast
dynamic_prediction = np.copy(data_m_cls[:len(data_m_cls) - test_size])

for i in range(len(data_m_cls) - test_size, len(data_m_cls)):
    last_feature = np.reshape(dynamic_prediction[i-time_window_m2:i], (1,time_window_m2,1))
    next_pred = model_m3.predict(last_feature)
    dynamic_prediction = np.append(dynamic_prediction, next_pred)

# dynamic prediction results, and transform back to the original scale
dynamic_prediction = dynamic_prediction.reshape(-1,1)
dynamic_prediction = scaler.inverse_transform(dynamic_prediction)
#%%
# visualization results
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls[:len(data_m_cls) - test_size]), label='Training Data')
plt.plot(np.arange(len(data_m_cls) - test_size, len(data_m_cls), 1), 
         scaler.inverse_transform(data_m_cls[-test_size:]), 
         label='Testing Data')
plt.plot(np.arange(len(data_m_cls) - test_size, len(data_m_cls), 1), 
         dynamic_prediction[-test_size:], 
         label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.show();
#%%
# MSE score
testScore = mean_squared_error(scaler.inverse_transform(data_m_cls[-test_size:]), 
                                         dynamic_prediction[-test_size:])
print('Dynamic Forecast MSE: {0:.2f}'.format(testScore))

#%%
origin = np.copy(data_m_cls)
prediction_m2 = origin
#%%
for i in range(len(prediction_m2)-5,len(prediction_m2)):
    if  i<=236:
          last_feature_m2= np.reshape(prediction_m2[-time_window_m2:len(prediction_m2)], (1,time_window_m2,1))
          next_pred_m2= model_m3.predict(last_feature_m2)
          prediction_m2 = np.append(prediction_m2, next_pred_m2)
    else:
        print(prediction_m2)
prediction_m2 = prediction_m2.reshape(-1,1)
prediction_m2 = scaler.inverse_transform(prediction_m2)

#%%
plt.figure()
plt.plot(scaler.inverse_transform(data_m_cls), label='Training Data')
plt.plot(np.arange(len(prediction_m2) - 5, len(prediction_m2), 1), prediction_m2[-5:], label='Out of Sample Prediction') 
plt.legend(loc = "upper left")
plt.title('5 month Forecasting results (ltsm)');
