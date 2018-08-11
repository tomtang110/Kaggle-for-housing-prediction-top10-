#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output #用于检测文件
# print(check_output(["ls","D:/adam/kaggle/house-price-pred/house-price-pred"]).decode("utf8")) #check the files available in the directory
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# print(train.head(5))

#check the numbers of samples and features
# print("The train data size before dropping Id feature is : {} ".format(train.shape))
# print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']
train_data = pd.read_csv('train.csv',index_col=0)
test_data = pd.read_csv('test.csv',index_col=0)
#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
# train.drop("Id", axis = 1, inplace = True)
# test.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
# print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
# print("The test data size after dropping Id feature is : {} ".format(test.shape))

# fig, ax = plt.subplots()
# ax.scatter(x = train['1stFlrSF'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('1stFlrSF', fontsize=13)
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('TotalBsmtSF', fontsize=13)
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x = train['YearRemodAdd'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('YearRemodAdd', fontsize=13)
# plt.show()


# #Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train.drop(train[(train['TotalBsmtSF']>4000)].index)

#Check the graphic again
# fig, ax = plt.subplots()
# ax.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(train['TotalBsmtSF'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('TotalBsmtSF', fontsize=13)
# plt.show()

# sns.distplot(train['SalePrice'] , fit=norm)

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm) #质量分布图

# Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')

# #Get also the QQ-plot
# # print(train['SalePrice'])
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

#连接数据
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

#missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(20))

#用直方图显示missing ratio
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#Correlation map to see how features are correlated with SalePrice
#关联性
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))#中位数
#可以算平均数

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#'MSZoning'里面'RL'最多 所以填RL
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#舍弃该列.因为在trainingset里全部为'ALLPUB',只有2个'NoSeWa'和1个'NA', 对预测影响不大
all_data = all_data.drop(['Utilities'], axis=1)

#NA means Typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#以下5个数据均只有一个NA, 所以填入最频繁的值
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

#NA = None
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
# print(missing_data.head())



#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#编号标准化
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
# print('Shape all_data: {}'.format(all_data.shape))

#add one important feature for area related features
#可考虑给每个部分根据关联度加权值
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
# print(skewness.head(10))


skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])

all_data = pd.get_dummies(all_data)
# print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]
X_train = train.values
X_test = test.values
y_value = y_train
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge,LassoLarsIC 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

n_folds=5
def fold_cv(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring="neg_mean_squared_error",cv=kf))
    return (rmse)

lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.001,random_state=1))
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.005,l1_ratio=0.9,random_state=3))
KRR= KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',
                                   min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
mode_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,gamma=0.0468,learning_rate=0.05,max_depth=3,min_child_weight=1.78,
                           n_estimators=2000,reg_alpha=0.464,reg_lambda=0.8671,subsample=0.5213,silent=1,
                           random_state=7,nthread=-1)

# lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0006,random_state=1))
# ENet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.0006,l1_ratio=0.8,random_state=3))
# KRR= KernelRidge(alpha=0.13,kernel='polynomial',degree=2,coef0=2.5)
# GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=3,max_features='sqrt',
#                                    min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
# mode_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,gamma=0.0468,learning_rate=0.05,max_depth=2,min_child_weight=1.78,
#                            n_estimators=2000,reg_alpha=0.464,reg_lambda=0.8671,subsample=0.5213,silent=1,
#                            random_state=7,nthread=-1)

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models=(ENet,GBoost,KRR),meta_model=lasso)
# score = fold_cv(stacked_averaged_models)
# print(1)
# print("\n meta_model stack:{:.4f},({:.4f})\n".format(score.mean(),score.std()))

def rmsle(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

stacked_averaged_models.fit(X_train,y_value)
stacked_train_pred = stacked_averaged_models.predict(X_train)
stacked_pre = np.expm1(stacked_averaged_models.predict(X_test))
print(rmsle(y_value,stacked_train_pred))

mode_xgb.fit(X_train,y_value)
xgb_train_pred = mode_xgb.predict(X_train)
xgb_pre = np.expm1(mode_xgb.predict(X_test))
print(rmsle(y_value,xgb_train_pred))

print('RMSLE score on train data:')
print(rmsle(y_value,stacked_train_pred*0.8 +
               xgb_train_pred*0.2))

ensemble = stacked_pre*0.7+xgb_pre*0.3

submission = pd.DataFrame(data={'Id':test_data.index,'SalePrice':ensemble})
submission.to_csv("submission.csv",index=False)
