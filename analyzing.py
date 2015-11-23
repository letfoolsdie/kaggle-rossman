# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:03:49 2015

@author: Nikolay_Semyachkin
"""

###USING MEDIAN VALUES FOR EACH GROUP GROUPDED BY DAYOFWEEK, STORE AND PROMO FOR PREDICTION###

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib
matplotlib.style.use('ggplot')

def rmspe(y, yhat):
    fr = pd.DataFrame(y)
    fr['Sales_pred'] = yhat
    fr.columns = ['Sales','Sales_pred']
    fr['res'] = ((fr.Sales-fr.Sales_pred)/fr.Sales)**2
    return np.sqrt(np.mean(fr.loc[fr.Sales != 0].res))

df = pd.read_csv('train.csv')
cols1 = df.columns

train_results = []
test_results = []
repeat = 10

for i in range(repeat):
    df, test = train_test_split(df, test_size = 0.2)
    df = pd.DataFrame(df, columns = cols1)
    test = pd.DataFrame(test, columns = cols1)
    
    df['Sales'] = df.Sales.astype(float)
    test['Sales'] = test.Sales.astype(float)
    
    df = df.loc[df.Sales>0]
    test.loc[ test.Open.isnull(), 'Open' ] = 1
    
    
    cols = ['DayOfWeek', 'Store','Promo']
    
    medians = df.groupby(cols)['Sales'].median()
    medians = medians.reset_index()
    medians['Sales_pred'] = medians['Sales']
    del medians['Sales']
    
    df1 = pd.merge(df,medians,on=cols,how = 'left')
    df1.loc[ df1.Open == 0, 'Sales_pred' ] = 0
    assert( df1.Sales_pred.isnull().sum() == 0 )
    train_error = rmspe(df1.Sales,df1.Sales_pred)
    print('train set error', train_error)
    
    test2 = pd.merge(test,medians,on=cols,how = 'left')
    assert( len( test2 ) == len( test ))
    
    test2.loc[ test2.Open == 0, 'Sales_pred' ] = 0
    assert( test2.Sales_pred.isnull().sum() == 0 )
    test_error = rmspe(test2.Sales,test2.Sales_pred)
    print('test set error',test_error)
    train_results.append(train_error)
    test_results.append(test_error)
