# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:24:46 2015

@author: Nikolay_Semyachkin
"""

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.dates 
import datetime 
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.cross_validation import train_test_split
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
store = pd.read_csv('store.csv')

print('Doing some preprocessing...')

datetimes = [datetime.datetime.strptime(t, "%Y-%m-%d") for t in train.Date]
plotData = matplotlib.dates.date2num(datetimes) 
train = train.join(pd.DataFrame(plotData,columns = ['datetimes']))
train = train.join(pd.DataFrame(train.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))

toAppend = pd.DataFrame(np.log(train.Sales+1),dtype=float)
toAppend.columns.values[0]='LogSale'
train=train.join(toAppend)

proInt = store.PromoInterval.apply(myPinterval).tolist()
proInt = pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)])
store = store.drop('PromoInterval',1).join(proInt)

store = store.drop('StoreType',1).join(pd.get_dummies(store['StoreType']).rename(columns=lambda x: 'StoreType' +"_"+str(x)))  
store = store.drop('Assortment',1).join(pd.get_dummies(store['Assortment']).rename(columns=lambda x: 'Assortment' +"_"+str(x)))

train.StateHoliday = [mychange(x) for x in train.StateHoliday]

#cols = train.columns
#train, test = train_test_split(train, test_size = 0.2)
#train = pd.DataFrame(train, columns = cols)
#test = pd.DataFrame(test, columns = cols)


train = train.drop('StateHoliday',1).join(pd.get_dummies(train['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x))) 
#newtest = test.drop('StateHoliday',1).join(pd.get_dummies(test['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x))) 

train=pd.merge(train, store, on="Store")  
#newtrain.drop(['Date','Customers','datetimes'],axis = 1,inplace=True)

#newtest=pd.merge(newtest, store, on="Store")  
#newtest.drop(['Date','Customers','datetimes','Sales'],axis = 1,inplace=True)
cols = train.columns
#assert(newtrain.columns == newtest.columns)
train_results = []
test_results = []
repeat = 5
print('Splitting data...')
for i in range(repeat):
    newtrain, newtest = train_test_split(train, test_size = 0.2)
    newtrain = pd.DataFrame(newtrain, columns = cols)
    newtest = pd.DataFrame(newtest, columns = cols)
    
    #test = test.join(pd.DataFrame(test.Date.apply(splitTime).tolist(), columns = ['year','mon','day']))
    #newtest = test.drop('StateHoliday',1).join(pd.get_dummies(test['StateHoliday']).rename(columns=lambda x: 'StateHoliday' +"_"+str(x)))  
    #newtest = pd.merge(newtest,store, on="Store")
    #newtest.drop(['Date'],axis = 1,inplace=True) 
    
    #assert(np.sum(newtrain.var()==0)==0)
    #
    #toDrop = list(set(newtrain.columns.values)-set(newtest.columns.values) )
    features = [col for col in newtrain.columns if col not in ['Customers', 'Sales', 'Date','LogSale','datetimes']]
    #
    rf = RandomForestRegressor(n_estimators=100)
    print('Starting training...')
    rf.fit(newtrain[features].fillna(-1),newtrain.LogSale)
    print('Predicting train values...')
    newtrain['mypred'] = rf.predict(newtrain[features].fillna(-1))
    newtrain['mypred'] = np.exp(newtrain['mypred'])-1
    train_error = rmspe(newtrain[newtrain.Sales>0].Sales,newtrain[newtrain.Sales>0].mypred)
    print('train set error',train_error)
    newtest['mypred'] = rf.predict(newtest[features].fillna(-1))
    newtest['mypred'] = np.exp(newtest['mypred'])-1
    test_error = rmspe(newtest[newtest.Sales>0].Sales,newtest[newtest.Sales>0].mypred)
    print('test set error',test_error)
    train_results.append(train_error)
    test_results.append(test_error)

print('mean train error', np.mean(train_results))
print('mean test error',np.mean(test_results))


