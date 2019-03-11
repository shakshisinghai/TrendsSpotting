# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:28:14 2019

@author: sir
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:12:50 2019

@author: sir
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from pandas import Series
from pyramid.arima import auto_arima

import pickle
#reading dataset
data = pd.read_csv('kaggelsample.csv')
col=list(data)
#data type here is object (month) Let’s convert it into a Time series object and use the Month column as our index.
        
from datetime import datetime
import pickle

con=data['Month']
data['Month']=pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
#check datatype of index
data.index

a={}

#convert to time series:
for (c) in col:
    if(c!='Month'):
    
        stepwise_model = auto_arima(data[c], start_p=1, start_q=1,
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=True,
                                   d=1, D=1, trace=True,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)
        print(stepwise_model.aic())
        train = data[c].loc['01/01/1993':'01/01/2015']
        stepwise_model.fit(train)
        with open("trendspot.pkl",'wb') as f:
            pickle.dump(stepwise_model, f)
        
        n=2
        future_forecast = stepwise_model.predict(n_periods=n)
      
        a[c]=future_forecast[n-1]
        
print (a)

trend=max(a, key=a.get)
print(trend, a[trend])
