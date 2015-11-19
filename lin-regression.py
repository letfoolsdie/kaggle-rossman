# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:42:18 2015

@author: Nikolay_Semyachkin
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy import stats
import matplotlib
matplotlib.style.use('ggplot')

def rmspe(y, yhat):
    fr = pd.DataFrame(y)
    fr['Sales_pred'] = yhat
    fr.columns = ['Sales','Sales_pred']
    fr['res'] = ((fr.Sales-fr.Sales_pred)/fr.Sales)**2
    return np.sqrt(np.mean(fr.loc[fr.Sales != 0].res))
    
df = pd.read_csv('train.csv')
train, test = train_test_split(df, test_size = 0.2)
train = pd.DataFrame(train, columns = df.columns)
test = pd.DataFrame(test, columns = df.columns)
#train['Sales'] = train.Sales.astype(float)
#test['Sales'] = test.Sales.astype(float)
#train['Promo'] = train.Promo.astype(int)
#test['Promo'] = test.Promo.astype(int)

train = train.loc[train.Sales > 0]
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(train.Promo, train.Sales)

train['Sales_pred'] = intercept + slope * train.Promo
train.loc[train.Open != 1, 'Sales'] = 0

