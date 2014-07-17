import numpy as np
import cPickle
import itertools

def get_comb_feat(): 
	with open('gini_val.pkl','rb') as gvfile:
		gini_val = cPickle.load(gvfile)
	gini_val = np.array(gini_val)
	feat_list = []
	pos_list = []
	gnum = 0
	for item_val in gini_val:
		gnum += 1
		if item_val > 0.10:
			print '|==> ',item_val,' Valpos is ',gnum
			feat_list.append(item_val)
			pos_list.append(gnum)
	#无序组合
	return list(itertools.combinations(pos_list,2))
