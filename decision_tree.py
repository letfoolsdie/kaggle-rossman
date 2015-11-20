# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:27:07 2015

@author: Nikolay_Semyachkin
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy import stats
import matplotlib
from sklearn.ensemble import RandomForestRegressor
matplotlib.style.use('ggplot')

def rmspe(y, yhat):
    fr = pd.DataFrame(y)
    fr['Sales_pred'] = yhat
    fr.columns = ['Sales','Sales_pred']
    fr['res'] = ((fr.Sales-fr.Sales_pred)/fr.Sales)**2
    return np.sqrt(np.mean(fr.loc[fr.Sales != 0].res))
    
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#df = df[:len(df)]
#train, test = train_test_split(df, test_size = 0.2)
#train = pd.DataFrame(train, columns = df.columns)
train = df
#test = pd.DataFrame(test, columns = df.columns)

features = ['Store', 'Open','DayOfWeek','Promo']

clf = RandomForestRegressor(n_estimators=150, min_samples_split=1)
y = train.Sales
clf.fit(train[features], y)
train['pred_sales'] = clf.predict(train[features])
print('train set error',rmspe(train[train.Sales>0].Sales,train[train.Sales>0].pred_sales))
test['pred_sales'] = clf.predict(test[features])
#print('test set error',rmspe(test[test.Sales>0].Sales,test[test.Sales>0].pred_sales))
test['Sales'] = test.pred_sales
test[[ 'Id', 'Sales' ]].to_csv('rand_for.csv', index = False )



