from __future__ import division
from sklearn import linear_model
#from sklearn import ensemble
#from sklearn import svm
#from sklearn import tree
#from sklearn import neighbors
from gini_func import *
import numpy as np
import pandas as pd

traincv = pd.read_csv('../datadir/train.csv')
testcv = pd.read_csv('../datadir/test.csv')
tr = traincv[['var10','var11','var13','var15']]
ts = testcv[['var10','var11','var13','var15']]
tr = np.nan_to_num(np.array(tr))
ts = np.nan_to_num(np.array(ts))
#print traincv.ix[:3,:2] first one is row second is column

#get the tr and ts array to be the train set and test set.

#1.Ridge Regression => 0.5 = 0.24..
clf = linear_model.Ridge(alpha=0.5)
clf.fit(tr,traincv['target'].values)
#2.Lasso Regression => 0.5 = 0.126..
#clf = linear_model.Lasso(alpha=0.5)
#clf.fit(tr,traincv['target'].values)
#3.RandomForest => 10 = 0.229.. 100 = 0.261..
#clf = ensemble.GradientBoostingRegressor(n_estimators=100)
#clf.fit(tr,traincv['target'].values)
#4.SVR need regularization!
#clf = svm.SVR()
#clf.fit(tr,traincv['target'].values)
#5.LogisticRegression also need regularization!
#clf = linear_model.LogisticRegression(C=1.5)
#clf.fit(tr,traincv['target'].values)
#6.DecisionTree 
#clf = tree.DecisionTreeRegressor()
#clf.fit(tr,traincv['target'].values
#7.KNN 
#clf = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distence')
#clf.fit(tr,traincv['target'].values


#we need act,pred,weights
pred = clf.predict(ts)
sample = pd.read_csv('../datadir/sampleSubmission.csv')
sample['target'] = pred
sample.to_csv('submission.csv',index=False)

#pred = (np.array(preds)  - np.array(testcv['var13'])) /2.0
#cact = pd.read_csv('cv_act.csv')
#weights = np.array(testcv['var11'])
#act = np.array(cact['target'])

#print normalized_weighted_gini(act,pred,weights)
