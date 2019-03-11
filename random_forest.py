# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:02:41 2019

@author: DESKTOP
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np
import statsmodels
import pickle
import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor



BUCKET_NAME = "random_forest_prediction"
MODEL_FILE = 'model.pkl'
data_file = 'Weekly_Sales.csv'
data_dir = 'gs://random_forest_prediction/Data'



subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    data_file),
                       data_file], stderr=sys.stdout)



def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))

data = pd.read_csv('Weekly_Sales.csv')
data = data.filter(regex=r'Product|W')
melt = data.melt(id_vars='Product_Code', var_name='Week', value_name='Sales')

melt['Product_Code'] = melt['Product_Code'].str.extract('(\d+)', expand=False).astype(int)

melt['Week'] = melt['Week'].str.extract('(\d+)', expand=False).astype(int)

melt = melt.sort_values(['Week', 'Product_Code'])
melt4 = melt.copy()

melt4['Last_Week_Sales'] = melt4.groupby(['Product_Code'])['Sales'].shift()
melt4['Last_Week_Diff'] = melt4.groupby(['Product_Code'])['Last_Week_Sales'].diff()
melt4['Last-1_Week_Sales'] = melt4.groupby(['Product_Code'])['Sales'].shift(2)
melt4['Last-1_Week_Diff'] = melt4.groupby(['Product_Code'])['Last-1_Week_Sales'].diff()
melt4['Last-2_Week_Sales'] = melt4.groupby(['Product_Code'])['Sales'].shift(3)
melt4['Last-2_Week_Diff'] = melt4.groupby(['Product_Code'])['Last-2_Week_Sales'].diff()
melt4 = melt4.dropna()


mean_error = []
week=melt["Week"].values[-1]
train = melt4[melt4['Week'] < week]
val = melt4[melt4['Week'] == week]

xtr, xts = train.drop(['Sales'], axis=1), val.drop(['Sales'], axis=1)
ytr, yts = train['Sales'].values, val['Sales'].values

mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
result=mdl.fit(xtr, ytr)


pipeline = Pipeline([
    ('classifier', result)
])
    

with open(MODEL_FILE, 'wb') as model_file:
  pickle.dump(pipeline, model_file)




gcs_model_path = os.path.join('gs://', BUCKET_NAME,datetime.datetime.now().strftime('randomforest_prediction_%Y%m%d_%H%M%S'), MODEL_FILE)

subprocess.check_call(['gsutil', 'cp', MODEL_FILE, gcs_model_path],stderr=sys.stdout)


"""

p = mdl.predict(xts)

error = rmsle(yts, p)
print('Week %d - Error %.5f' % (week, error))
mean_error.append(error)
print('Mean Error = %.5f' % np.mean(mean_error))

np.argmax(p)
"""