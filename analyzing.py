# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:03:49 2015

@author: Nikolay_Semyachkin
"""

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
df, test = train_test_split(df, test_size = 0.2)
df = pd.DataFrame(df, columns = cols1)
df['Sales'] = df.Sales.astype(float)

test = pd.DataFrame(test, columns = cols1)
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
print('train set error',rmspe(df1.Sales,df1.Sales_pred))

test2 = pd.merge(test,medians,on=cols,how = 'left')
assert( len( test2 ) == len( test ))

test2.loc[ test2.Open == 0, 'Sales_pred' ] = 0
assert( test2.Sales_pred.isnull().sum() == 0 )
print('test set error',rmspe(test2.Sales,test2.Sales_pred))


#test2[[ 'Id', 'Sales' ]].to_csv('output_file.csv', index = False )
#
#df['Date'] = pd.to_datetime(df.Date)
#cols = df.columns
#
#train, test = train_test_split(df, test_size = 0.2)
#train = pd.DataFrame(train, columns = cols)
#test = pd.DataFrame(test, columns = cols)
#
#train.to_csv('train_loc.csv', index=False)
#test.to_csv('test_loc.csv', index=False)
