import numpy as np
import pandas as pd
import warnings 
# def ignore_warn(*args, **kwargs):
#     pass
# warnings.warn = ignore_warn

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train_data = pd.read_csv('train.csv',index_col=0)
test_data = pd.read_csv('test.csv',index_col=0)
y_train = np.log1p(train_data.pop('SalePrice'))
data = pd.concat(((train_data,test_data)),axis=0)
data['MSSubClass']=data['MSSubClass'].astype(str)
data_dummy = pd.get_dummies(data)
data_dummy_mean = data_dummy.mean()
data_dummy = data_dummy.fillna(data_dummy_mean)
data_numeric = data.columns[data.dtypes != 'object']
data_dummy_nu_mean=data_dummy.loc[:,data_numeric].mean()
data_dummy_nu_sd = data_dummy.loc[:,data_numeric].std()
data_dummy.loc[:,data_numeric] = (data_dummy.loc[:,data_numeric]-data_dummy_nu_mean)/data_dummy_nu_sd
train_dummy=data_dummy.loc[train_data.index]
test_dummy = data_dummy.loc[test_data.index]

X_train = train_dummy.values
X_test = test_dummy.values
y_values = y_train.values
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
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train_dummy.values)
    rmse = np.sqrt(-cross_val_score(model,X_train,y_values,scoring="neg_mean_squared_error",cv=kf))
    return (rmse)

lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.001,random_state=1))
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.005,l1_ratio=0.9,random_state=3))
KRR= KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000,learning_rate=0.05,max_depth=4,max_features='sqrt',
                                   min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=5)
mode_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,gamma=0.0468,learning_rate=0.05,max_depth=3,min_child_weight=1.78,
                           n_estimators=2000,reg_alpha=0.464,reg_lambda=0.8671,subsample=0.5213,silent=1,
                           random_state=7,nthread=-1)
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
        
                print(y[train_index])
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
score = fold_cv(stacked_averaged_models)

print("\n meta_model stack:{:.4f},({:.4f})\n".format(score.mean(),score.std()))



