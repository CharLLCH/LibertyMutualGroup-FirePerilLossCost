from __future__ import division
# -*- coding=utf-8 -*-
import numpy as np
import pandas as pd
from prehand import *
from gini_func import *
from sklearn import linear_model
import cPickle

#train every feature at once or store and compute?
#I choose store.because store for once could use easily.
#of course,into the csv line by row.
print '|==> START TO RECONSTRUCT THE TRAIN AND TEST SET <==|'
#handler the train set.
trainset = pd.read_csv('../datadir/cv_train.csv')
print '|==> train set is loaded <==|'
testset = pd.read_csv('../datadir/cv_test.csv')
print '|==> test set is loaded <==|'
#get the train weight and target.
weights = trainset['var11']
targets = trainset['target']
with open('../datadir/weight_train.pkl','wb',2) as wtfile:
	cPickle.dump(weights,wtfile)
wtfile.close()
with open('../datadir/target_train.pkl','wb',2) as wtfile:
	cPickle.dump(targets,wtfile)
wtfile.close()
print '|==> weights vect is loaded <==|'
#should confirm the dimension of each feature.
print '|==> start to get the dim_list. <==|'
num_trainfeat_dim = []
i_fe_num = 1
for each_feat in trainset.ix[:,2:10]:
	num_trainfeat_dim.append(num_dm_feat(trainset[each_feat],i_fe_num))
	i_fe_num += 1
num_trainfeat_dim.append(num_dm_feat(trainset['dummy'],i_fe_num))
num_testfeat_dim = []
i_fe_num = 1
for each_feat in testset.ix[:,1:9]:
	num_testfeat_dim.append(num_dm_feat(testset[each_feat],i_fe_num))
	i_fe_num += 1
num_testfeat_dim.append(num_dm_feat(testset['dummy'],i_fe_num))
fdim_list = final_dim_list(num_trainfeat_dim,num_testfeat_dim)

print '|==> start to reconstruct the feat <==|'
num_var = 0
train_vect = []
#restruct the var_feature in turn.
for feat_val in trainset.ix[:,2:]:
	var_vect = []
	num_var += 1
	if num_var < 10 or num_var == 18:
		if num_var == 4:
			var_vect = feat_azn(trainset[feat_val],fdim_list[num_var-1])
		elif num_var != 18:
			var_vect = feat_az19(trainset[feat_val],fdim_list[num_var-1])
		else:
			var_vect = feat_az19(trainset[feat_val],fdim_list[-1])
	else:
		var_vect = feat_num(trainset[feat_val])
	train_vect.append(var_vect)
	print '|==> ',feat_val,'in train is reconstructed <==|'
with open('../datadir/nor_feat_train.pkl','wb') as ffile:
	cPickle.dump(train_vect,ffile,2)
ffile.close()
train_vect = []
trainset = []
	
#handler the test vect.
weights = testset['var11']
with open('../datadir/weight_test.pkl','wb',2) as wtfile:
	cPickle.dump(weights,wtfile)
wtfile.close()
print '|==> weights vect is loaded <==|'
num_var = 0
test_vect = []
print '|==> start to reconstruct the feat <==|'
#restruct the var_feature in turn.
for feat_val in testset.ix[:,1:]:
	var_vect = []
	num_var += 1
	if num_var < 10 or num_var == 18:
		if num_var == 4:
			var_vect = feat_azn(testset[feat_val],fdim_list[num_var-1])
		elif num_var != 18:
			var_vect = feat_az19(testset[feat_val],fdim_list[num_var-1])
		else:
			var_vect = feat_az19(testset[feat_val],fdim_list[-1])
	else:
		var_vect = feat_num(testset[feat_val])
	test_vect.append(var_vect)
	print '|==> ',feat_val,'in test is reconstructed <==|'
with open('../datadir/nor_feat_test.pkl','wb') as ffile:
	cPickle.dump(test_vect,ffile,2)
ffile.close()
test_vect = []
testset = []
