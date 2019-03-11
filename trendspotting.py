# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:12:50 2019

@author: sir
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from pandas import Series
from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf  

from pyramid.arima import auto_arima
import pickle
#reading dataset
data = pd.read_csv('product_sold.csv')
col=list(data)
print (data.head())
print ('\n Data Types:')
print (data.dtypes)

#data type here is object (month) Let’s convert it into a Time series object and use the Month column as our index.
        
from datetime import datetime
con=data['Month']
data['Month']=pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
#check datatype of index
data.index

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)
a={}

#convert to time series:
for (c) in col:
    if(c!='Month'):
        
        ts =data[c]
        plt.plot(ts)
        ts_log = np.log(ts)
        #plt.plot(ts_log)
        
        
        moving_avg = ts_log.rolling(12).mean()
        #plt.plot(ts_log)
        #plt.plot(moving_avg, color='red')
        
        ts_log_moving_avg_diff = ts_log - moving_avg
        ts_log_moving_avg_diff.head(12)
        
        ts_log_moving_avg_diff.dropna(inplace=True)
        
        test_stationarity(ts_log_moving_avg_diff)
        
        
        #SEASONALITY (ALONG WITH TREND)
        #Take first difference:
        ts_log_diff = ts_log - ts_log.shift()
        #plt.plot(ts_log_diff)
        
        ts_log_diff.dropna(inplace=True)
        test_stationarity(ts_log_diff)
        
        
        lag_acf = acf(ts_log_diff, nlags=12)
        lag_pacf = pacf(ts_log_diff, nlags=12, method='ols')
        
        model=ARIMA(ts_log, order=(2, 1, 2))
        result_AR= model.fit(disp=-1)
        #plt.plot(ts_log_diff)
        #plt.plot(result_AR.fittedvalues, color='red')
        
        pre_ARIMA_diff=pd.Series(result_AR.fittedvalues, copy=True)
        print(pre_ARIMA_diff.head())
        
        pre_ARIMA_diff_cumsum=pre_ARIMA_diff.cumsum()
        print(pre_ARIMA_diff_cumsum.head())
        
        pre_ARIMA_log=pd.Series(ts_log.ix[0], index=ts_log.index)
        pre_ARIMA_log=pre_ARIMA_log.add(pre_ARIMA_diff_cumsum, fill_value=0)
        pre_ARIMA_log.head()
        
        pre_ARIMA=np.exp(pre_ARIMA_log)
        #plt.plot(ts)
        #plt.plot(pre_ARIMA)
        pre_ARIMA.head()
         
        #pre_ARIMA['1961-01-01']
        
        stepwise_model = auto_arima(data[c], start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=True,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)
        print(stepwise_model.aic())
        
        train = data[c].loc['1950-01-01':'2000-12-01']
        stepwise_model.fit(train)
        n=20
        future_forecast = stepwise_model.predict(n_periods=n)
      
        a[c]=future_forecast[n-1]
        
print (a)

trend=max(a, key=a.get)
print(trend, a[trend])
# sav/e the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
