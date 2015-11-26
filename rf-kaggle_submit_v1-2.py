# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:24:46 2015

@author: Nikolay_Semyachkin
"""
###THIS SUBMISSION GIVES 0.12251 ON PUBLIC LEADERBOARD
##In version 1-2: change dummy variables to simply different numbers in one columns
##Your submission scored 0.12364

import pandas as pd
import numpy as np  
from sklearn.ensemble.forest import RandomForestRegressor
#%matplotlib inline


def rmspe(y, yhat):
    fr = pd.DataFrame(y)
    fr['Sales_pred'] = yhat
    fr.columns = ['Sales','Sales_pred']
    fr['res'] = ((fr.Sales-fr.Sales_pred)/fr.Sales)**2
    return np.sqrt(np.mean(fr.loc[fr.Sales != 0].res))
    
        
def processdata(data):
    data.loc[data.Open.isnull(), 'Open'] = 1
    
    data['year'] = data.Date.apply(lambda x: x.split('-')[0])
    data['year'] = data['year'].astype(float)
    data['mon'] = data.Date.apply(lambda x: x.split('-')[1])
    data['mon'] = data['mon'].astype(float)
    data['day'] = data.Date.apply(lambda x: x.split('-')[2])
    data['day'] = data['day'].astype(float)

    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'
    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'
    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'
    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'
    data['StoreType'] = data['StoreType'].astype(float)
    
    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'
    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'
    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'
    data['Assortment'] = data['Assortment'].astype(float)

    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'
    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'
    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'
    data['StateHoliday'] = data['StateHoliday'].astype(float)
    
    data.loc[data['PromoInterval'] == 'Feb,May,Aug,Nov', 'PromoInterval'] = '1'
    data.loc[data['PromoInterval'] == 'Jan,Apr,Jul,Oct', 'PromoInterval'] = '2'
    data.loc[data['PromoInterval'] == 'Mar,Jun,Sept,Dec', 'PromoInterval'] = '3'
    data['PromoInterval'] = data['PromoInterval'].astype(float)
    
    data.fillna(0, inplace=True)
    
print('Loading data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

print('Doing some preprocessing...')

train['LogSale'] = np.log(train.Sales+1)

train=pd.merge(train, store, on="Store")  
test = pd.merge(test, store, on="Store")

processdata(train)
processdata(test)


repeat = 1
#print('Splitting data...')
for i in range(repeat):
    features = [col for col in test.columns if col not in ['Customers', 'Sales', 'Date','LogSale','datetimes','Id']]
    rf = RandomForestRegressor(n_estimators=100)
    print('Starting training...')
    rf.fit(train[features],train.LogSale)
    train['mypred'] = rf.predict(train[features])
    train['mypred'] = np.expm1(train.mypred)
    train_error = rmspe(train[train.Sales>0].Sales,train[train.Sales>0].mypred)
    
#    
    test['mypred'] = rf.predict(test[features])
    test['mypred'] = np.exp(test['mypred'])-1
#
test['Sales'] = test.mypred
test[[ 'Id', 'Sales' ]].to_csv('rand_for_kag_v4-7_not_dummy.csv', index = False )



