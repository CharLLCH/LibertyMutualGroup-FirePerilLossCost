from __future__ import division
from sklearn import linear_model
#from sklearn import ensemble
#from sklearn import svm
#from sklearn import tree
#from sklearn import neighbors
from gini_func import *
import numpy as np
import pandas as pd
import cPickle

#print traincv.ix[:3,:2] first one is row second is column
#1.Ridge Regression 
#clf = linear_model.Ridge(alpha=0.5)
#clf.fit(tr,traincv['target'].values)
#2.Lasso Regression
#clf = linear_model.Lasso(alpha=0.5)
#clf.fit(tr,traincv['target'].values)
#3.RandomForest
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

#get the cv_train and cv_test array to be the train set and test set.
#there will be the existed file to read.
print '|==> Start To Train The Model And Predict. <==|'
with open('../datadir/nor_feat_train.pkl','rb') as cvt_file:
	cv_train = cPickle.load(cvt_file)
cvt_file.close()
print '|==> CV_TRAIN is loaded. <==|'
with open('../datadir/weight_test.pkl','rb') as wtfile:
	tmp_w = cPickle.load(wtfile)
	weights_test = np.array(map(float,tmp_w))
wtfile.close()
print '|==> Weights of cv_test is loaded. <==|'
with open('../datadir/target_train.pkl','rb') as wtfile:
	tmp_t = cPickle.load(wtfile)
	cv_target = np.array(map(float,tmp_t))
wtfile.close()
with open('../datadir/nor_feat_test.pkl','rb') as cvt_file:
	cv_test = cPickle.load(cvt_file)
cvt_file.close()
print '|==> CV_TEST is loaded. <==|'
cact = pd.read_csv('../datadir/cv_act.csv')
act = np.array(cact['target'])
gini_list = []
var_num = 0
#cv_train or cv_test is the regularized list.
#train the single feat in turn.
n = len(cv_train)
for i_f in xrange(n):
	one_feat = cv_train[i_f]
	o_f = cv_test[i_f]
	var_num += 1
	if i_f < 9 or i_f == 17:
		feat_one = np.array(one_feat)
		f_o = np.array(o_f)
	else:
		feat_one = np.nan_to_num(np.array(one_feat).reshape(len(one_feat),1))
		f_o = np.nan_to_num(np.array(o_f).reshape(len(o_f),1))
	clf = linear_model.Ridge(alpha = 0.5)
	print np.shape(feat_one),np.shape(cv_target)
	clf.fit(feat_one,cv_target)
	print np.shape(feat_one),'<train===test>',np.shape(f_o)
	#we need act,pred,weights
	pred = clf.predict(f_o)
	print '|==> No.',var_num,' has train and predicted.<==|'
	tmp_gini = normalized_weighted_gini(act,pred,weights_test)
	print 'No.',var_num,'\'s gini is ',tmp_gini
	gini_list.append(tmp_gini)
with open('gini_val.pkl','wb') as gfile:
	cPickle.dump(gini_list,gfile,2)
gfile.close()
print 'all single feature has trained and predicted.'
