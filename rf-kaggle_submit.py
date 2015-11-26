# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:24:46 2015

@author: Nikolay_Semyachkin
"""
###THIS SUBMISSION GIVES 0.12251 ON PUBLIC LEADERBOARD

import pandas as pd
import numpy as np  
import matplotlib.dates 
import datetime 
from sklearn.ensemble.forest import RandomForestRegressor
#%matplotlib inline


def rmspe(y, yhat):
    fr = pd.DataFrame(y)
    fr['Sales_pred'] = yhat
    fr.columns = ['Sales','Sales_pred']
    fr['res'] = ((fr.Sales-fr.Sales_pred)/fr.Sales)**2
    return np.sqrt(np.mean(fr.loc[fr.Sales != 0].res))
    
    
def mychange(x):
     if type(x)!= str: x=str(x)
     return x
     
     
def splitTime(x): 
    mysplit = datetime.datetime.strptime(x,  "%Y-%m-%d") 
    return [mysplit.year,mysplit.month,mysplit.day]


def myPinterval(x):
    if x=='Feb,May,Aug,Nov':  return([0,1,0,0,1,0,0,1,0,0,1,0])
    elif x=='Jan,Apr,Jul,Oct':  return([1,0,0,1,0,0,1,0,0,1,0,0])
    elif x== 'Mar,Jun,Sept,Dec': return([0,0,1,0,0,1,0,0,1,0,0,1])
    else: return(np.repeat(0,12).tolist())
    
print('Loading data...')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

print('Doing some preprocessing...')

datetimes = [datetime.datetime.strptime(t, "%Y-%m-%d") for t in train.Date]

plotData = matplotlib.dates.date2num(datetimes) 
#train = train.join(pd.DataFrame(plotData,columns = ['datetimes']))
train = train.join(pd.DataFrame(train.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))

test = test.join(pd.DataFrame(test.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))

train['LogSale'] = np.log(train.Sales+1)

proInt = store.PromoInterval.apply(myPinterval).tolist()
proInt = pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)])
store = store.drop('PromoInterval',1).join(proInt)
store = store.drop('StoreType',1).join(pd.get_dummies(store['StoreType']).rename(columns=lambda x: 'StoreType' +"_"+str(x)))  
store = store.drop('Assortment',1).join(pd.get_dummies(store['Assortment']).rename(columns=lambda x: 'Assortment' +"_"+str(x)))

train['StateHoliday'] = [mychange(x) for x in train.StateHoliday]
test['StateHoliday'] = [mychange(x) for x in test.StateHoliday]

train = train.drop('StateHoliday',1).join(pd.get_dummies(train['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x))) 
test = test.drop('StateHoliday',1).join(pd.get_dummies(test['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x)))  

train=pd.merge(train, store, on="Store")  
test = pd.merge(test, store, on="Store")

repeat = 1
print('Splitting data...')
for i in range(repeat):
    features = [col for col in test.columns if col not in ['Customers', 'Sales', 'Date','LogSale','datetimes','Id']]
    rf = RandomForestRegressor(n_estimators=100)
    print('Starting training...')
    rf.fit(train[features].fillna(-1),train.LogSale)

    test['mypred'] = rf.predict(test[features].fillna(-1))
    test['mypred'] = np.exp(test['mypred'])-1

test['Sales'] = test.mypred
test[[ 'Id', 'Sales' ]].to_csv('rand_for_kag_v4-5_confirm.csv', index = False )


