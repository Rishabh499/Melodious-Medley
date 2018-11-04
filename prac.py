import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
train=pd.read_csv('Desktop/dsg/train.csv')
test=pd.read_csv('Desktop/dsg/test.csv')
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

print train.columns

group=train.groupby("context_type")['is_listened']
mean=group.mean()

groupa=mean[mean>0.76].index
groupb=mean[mean<.37].index
def groupfun(x):
    if x in groupa:
        return 0
    if x in groupb:
        return 1
    else:
        return 2
train['group']=train.context_type.apply(groupfun)
test['group']=test.context_type.apply(groupfun)


group1=train.groupby("user_id")['is_listened']
mean1=group.mean()

groupa1=mean1[mean1>.8].index
groupb1=mean1[mean1<.3].index
def groupfun1(x):
    if x in groupa1:
        return 0
    if x in groupb1:
        return 1
    else:
        return 2
train['group1']=train.context_type.apply(groupfun1)
test['group1']=test.context_type.apply(groupfun1)

pg=test.copy()
def hello1(x):
    return (x[11:13])
for i in [train,test]:
    i['hour']=i.ts_listen.apply(hello1)
    i['hour']=i.hour.astype('int')

"""
sns.barplot(train.arti,train.is_listened)
plt.show()# barplot for hour
"""

def hello(x):
    return pd.to_datetime(x[:10])
for i in [train,test]:
    i.ts_listen=i.ts_listen.apply(hello)
    """i['year']=i.ts_listen.dt.year"""
    i['month']=i.ts_listen.dt.month
    i['day']=i.ts_listen.dt.day

for i in [train,test]:
    i.release_date=i.release_date.astype('str')
    def hell(x):
        return x[:4]
    i.ix[:,'release_date1']=i.release_date.apply(hell)
    i.release_date1=i.release_date1.astype('int')
    def hell1(x):
        return x[4:6]
    i.ix[:,'release_date2']=i.release_date.apply(hell1)
    i.release_date2=i.release_date2.astype('int')
    def hell2(x):
        return x[6:]
    i.ix[:,'release_date3']=i.release_date.apply(hell2)
    i.release_date3=i.release_date3.astype('int')
train.drop(["ID","ts_listen",'release_date'], axis = 1, inplace = True)
test.drop(["ID","ts_listen",'release_date'], axis = 1, inplace = True)
for i in [train,test]:
    """i['a']=i.year-i.release_date1"""
    i['c']=i.day-i.release_date3
    i['b']=i.month-i.release_date2

def timing(x):
    if x<=27:
        return 1
    else:
        return 0

train['half']=train.user_age.apply(timing)
test['half']=test.user_age.apply(timing)
train.media_duration=np.log1p(train.media_duration)
test.media_duration=np.log1p(test.media_duration)

"""
xtrain,xtest,ytrain,ytest=train_test_split(train.ix[:,train.columns!='is_listened'],train.ix[:,'is_listened'])
print xtrain.columns
xg=LGBMClassifier().fit(xtrain,ytrain)
print xg.feature_importances_
"""
poly=PolynomialFeatures(degree=2).fit(train.ix[:,train.columns!='is_listened'])
xtrain=poly.transform(train.ix[:,train.columns!='is_listened'])
xtest=poly.transform(test)
select=SelectPercentile(percentile=25).fit(xtrain,train.ix[:,'is_listened'])
xtrain=select.transform(xtrain)
xtest=select.transform(xtest)

"""
xtrain1,xtest1,ytrain1,ytest1=train_test_split(xtrain,train.ix[:,'is_listened'])
lr=LogisticRegression(C=.001).fit(xtrain1,ytrain1)
print lr.score(xtest1,ytest1)
"""

xg=LGBMClassifier(learning_rate=.1,num_iterations=600,max_depth=6).fit(xtrain,train.ix[:,'is_listened'])

print "hello"

pred1=xg.predict(xtest)


soln={"ID":pg["ID"],"to_listened":pred1}
soln=pd.DataFrame(soln)


soln.to_csv("solution.csv",index=False)


