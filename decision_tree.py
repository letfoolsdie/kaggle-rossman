# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:27:07 2015

@author: Nikolay_Semyachkin
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
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
print('Loading data...')

df = pd.read_csv('train.csv')
stores = pd.read_csv('store.csv')
df = pd.merge(df,stores,on='Store',how = 'left')
df = df.loc[df.Sales>0]

##Changing categorical numbers to values:
converting = preprocessing.LabelEncoder()
df['StateHoliday'] = converting.fit_transform(df.StateHoliday.astype(str))
df['Assortment'] = converting.fit_transform(df.Assortment)
df['StoreType'] = converting.fit_transform(df.StoreType)


df['logSales'] = np.log1p(df.Sales.astype(int))
#features = [col for col in df.columns if col not in ['Customers', 'Sales', 'Date','logSales','Promo2SinceWeek',
#       'Promo2SinceYear', 'PromoInterval','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','CompetitionDistance']]
#df = df[df.Sales>0]
#test = pd.read_csv('test.csv')
features = ['Store', 'Open','DayOfWeek','Promo']
#df = df[:len(df)]
print('Splitting data in train and test datasets...')
train_results = []
test_results = []
repeat = 10
##FOR THESE FEATURES (av per 10 repetitions):
#features = [col for col in df.columns if col not in ['Customers', 'Sales', 'Date','logSales','Promo2SinceWeek',

#np.mean(train_results)
#Out[115]: 0.20919888978361806
#np.mean(test_results)
#Out[116]: 0.23938207858573538

#FOR THESE BASIC FEATURES:
#features = ['Store', 'Open','DayOfWeek','Promo']
#np.mean(train_results)
#Out[118]: 0.21967295421005811
#
#np.mean(test_results)
#Out[119]: 0.21589105266903053

for i in range(repeat):
    train, test = train_test_split(df, test_size = 0.2)
    train = pd.DataFrame(train, columns = df.columns)
    #train = df
    test = pd.DataFrame(test, columns = df.columns)
    
    #features = ['Store', 'Open','DayOfWeek','Promo']
    #df2 = df[cols]
    
    print('Starting training...')
    clf = RandomForestRegressor(n_estimators=100, min_samples_split=1)
    y = train.logSales
    clf.fit(train[features], y)
    print('Training completed... \nPredicting test values...')
    train['pred_sales'] = clf.predict(train[features])
    train['pred_sales'] = np.expm1(train.pred_sales)
    train_error = rmspe(train[train.Sales>0].Sales,train[train.Sales>0].pred_sales)
    print('train set error',train_error)
    test.loc[test.Open.isnull(), 'Open'] = 1
    test['pred_sales'] = clf.predict(test[features])
    test['pred_sales'] = np.expm1(test.pred_sales)
    test_error = rmspe(test[test.Sales>0].Sales,test[test.Sales>0].pred_sales)
    print('test set error',test_error)
    train_results.append(train_error)
    test_results.append(test_error)
#test['Sales'] = test.pred_sales
#test[[ 'Id', 'Sales' ]].to_csv('rand_for_v2-2.csv', index = False )



