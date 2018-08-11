import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# use PANDAS to load data, index_col delegates the x column should be used 
# as indexing data which can not be utilized in processing.
train_data = pd.read_csv('train.csv',index_col=0)
test_data = pd.read_csv('test.csv',index_col=0)
# show first 5 row data.

# print(train_data.head())


prices = pd.DataFrame({"price":train_data["SalePrice"],"log(price+1)":np.log1p(train_data["SalePrice"])})
# print(prices)
# prices.hist()
# plt.show()

# remove y and concat test data and train data and log (p+1) both them;
train_y = np.log1p(train_data.pop('SalePrice'))
data = pd.concat(((train_data,test_data)),axis=0)
# identify data type 

# print(data['MSSubClass'].dtypes)

# change the number in MSSubClass into str as int or float may cause mistake
data['MSSubClass'] = data['MSSubClass'].astype(str)

# print(data['MSSubClass'].value_counts())
# print(pd.get_dummies(data['MSSubClass'],prefix='MSSubClass').head())

# one-Hot all data
data_dummy = pd.get_dummies(data)
# dealing with numerical data;
# show the number of NA in feature.

# print(data_dummy.isnull().sum().sort_values(ascending=False).head(10))

#compute the mean in  each numerical colum
data_mean = data_dummy.mean()

# print(data_mean.head(10))

#fill the NA with data_mean
data_dummy = data_dummy.fillna(data_mean)

# print(data_dummy.isnull().sum().sum())

#choose the numeric column
data_numeric = data.columns[data.dtypes != 'object']

# print(data_numeric)

# calculate the standard distribution and make data more smooth
# we also could use log(), but there are more ways to do 
# (x-x')/(sigma)
data_numeric_mean = data_dummy.loc[:,data_numeric].mean()
data_numeric_std = data_dummy.loc[:,data_numeric].std()
data_dummy.loc[:,data_numeric] = (data_dummy.loc[:,data_numeric]-data_numeric_mean)/data_numeric_std

#establish model

#reassign standardation data
train_dummy = data_dummy.loc[train_data.index]
test_dummy = data_dummy.loc[test_data.index]

# print(train_dummy.shape,test_dummy.shape)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#Change DF into numpy array
X_train = train_dummy.values

X_test = test_dummy.values

from sklearn.linear_model import Ridge
rid = Ridge(15)

from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

#Ridgebagging
# parameters = [1,10,15,20,25,30,40]
# test_scores=[]
# for para in parameters:
#     clf = BaggingRegressor(n_estimators=para,base_estimator=rid)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,scoring=
#     'neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))

# plt.plot(parameters,test_scores)
# plt.title('n_estimators VS CV error')
# plt.show()
#get answer eatimator = 20 and error = 0.132

#pure decision tree
# test_scores=[]
# parameters = [1,10,15,20,25,30,40]
# for para in parameters:
#     clf = BaggingRegressor(n_estimators=para)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,scoring=
#     'neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))

# plt.plot(parameters,test_scores)
# plt.title('n_estimators VS CV error')
# plt.show()
#get answer estimator = 15 and error =0.142
# from sklearn.ensemble import AdaBoostRegressor
# paras=[1,3,5,7,9]
# test_scores=[]
# for para in paras:
#     clf = AdaBoostRegressor(n_estimators=para,base_estimator=rid)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,
#     scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(paras,test_scores)
# plt.title('n_estimators VS CV error')
# plt.show()
#异常不稳定且效果不好.

from xgboost import XGBRegressor
# paras=[1,2,3,4,5,6]
# test_scores=[]
# for para in paras:
#     clf = XGBRegressor(max_depth=para)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,
#     scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(paras,test_scores)
# plt.title('max_depth VS CV error')
# plt.show()

#very good , error is decreased to 0.127 and max_depth is 5
xg = XGBRegressor(6)
xg.fit(X_train,train_y)
y_xg = np.expm1(xg.predict(X_test))

submission = pd.DataFrame(data={'Id':test_data.index,'SalePrice':y_xg})

# print(submission)
submission.to_csv("submission.csv",index=False)
# print(submission)


