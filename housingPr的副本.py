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

# print(data_dummy.isnull().sum().sum())d

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

# alphas = np.logspace(-3,2,50)
# test_scores = []
# for alpha in alphas:
#     clf = Ridge(alpha)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,
#     scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))

# plt.plot(alphas,test_scores)
# plt.title("Aphla vs CV Error")
# plt.show()

#random forest regressor
from sklearn.ensemble import RandomForestRegressor

# max_features = [.1,.3,.5,.7,.9,.99]
# test_scores=[]
# for max_f in max_features:
#     clf = RandomForestRegressor(n_estimators=200,max_features=max_f)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,train_y,cv=10,
#     scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))

# plt.plot(max_features,test_scores)
# plt.title("Max Features vs CV Error")
# plt.show()

rid = Ridge(15)
rft = RandomForestRegressor(n_estimators=500,max_features=.3)
rid.fit(X_train,train_y)
rft.fit(X_train,train_y)
y_rid = np.expm1(rid.predict(X_test))
y_rft = np.expm1(rft.predict(X_test))
y_res = (y_rid+y_rft)/2

submission = pd.DataFrame(data={'Id':test_data.index,'SalePrice':y_res})
print(submission)
















