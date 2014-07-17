from __future__ import division
import numpy as np

# -*- coding=utf-8 -*-

#get the single feature.
#note that,var1-var9 and dummy => Z NA Z*.
#divide by the type of the item in the feature.

indxdo = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
indxdt = {'A1':0,'B1':1,'C1':2,'D1':3,'D2':4,'D3':5,'D4':6,'D5':7,'E1':8,'E2':9,'E3':10,'E4':11,'E5':12,'E6':13,'F1':14,'G1':15,'G2':16,'H1':17,'H2':18,'H3':19,'I1':20,'J1':21,'J2':22,'J3':23,'J4':24,'J5':25,'J6':26,'K1':27,'L1':28,'M1':29,'N1':30,'O1':31,'O2':32,'P1':33,'Q1':34,'Q2':35,'Q3':36,'Q4':37,'Q5':38,'Q6':39,'Q7':40,'Q8':41,'R1':42,'YY':43}

#0.check the dimension.
#'0'=>48 'A'=>65
def num_dm_feat(featlist,n):
	set_char = set()
	is_z = 0
	if n != 4:
		for items in featlist:
			if items == 'Z':
				is_z = 1	
			else:
				tmpitem = indxdo[items]+1
				set_char.add(tmpitem)
	#here handler the Z type.
		tmpv = np.array(list(set_char)).max()
		#if tmpv > 9:
			#tmpv = tmpv - 16
		if is_z:
			tmpv = tmpv + 1
	else:
		tmpv = 44
	return tmpv

#get the final dim_list.
def final_dim_list(trdim_list,tedim_list):
	len_n = len(trdim_list)
	max_list = []
	for i_th in xrange(len_n):
		max_list.append(trdim_list[i_th] if trdim_list[i_th] >= tedim_list[i_th] else tedim_list[i_th])
	max_list[3] = 44
	return max_list
	
#1.A-Z and 1-9 type.
def feat_az19(featlist,ndim):
	feat_vect = []
	for item_feat in featlist:
		#for Z or any other missing value add one dimension.
		tmpvect = map(int,np.zeros(ndim))
		#still use dict to get the index.
		if item_feat == 'Z':
			tmpvect[ndim-1] = 1
		else:
			#print "length:",len(tmpvect),"dm_feat:",dm_feat,"items:",item_feat,indxdo[item_feat]
			tmpvect[indxdo[item_feat]] = 1
		feat_vect.append(tmpvect)
	return np.array(feat_vect)

#2.A1-R1 type.need rechange.
def feat_azn(featlist,ndim):
	feat_vect = []
	for item_feat in featlist:
		tmpvect = map(int,np.zeros(ndim))
		if item_feat in indxdt:
			tmpvect[indxdt[item_feat]] = 1
		else:
			tmpvect[indxdt['YY']] = 1
		feat_vect.append(tmpvect)
	return np.array(feat_vect)

#2-3.get the max and mean to handler the regularization and NA.
def max_mean_feat(featlist):
	tmpmax = 0
	tmpmean = 0
	tmpsum = 0
	for items in featlist:
		if items == 'NA':
			#the compensation is add the max,later can adjust.
			tmpsum += tmpmax
		else:
			fl_item = float(items)
			tmpmax = tmpmax if tmpmax > fl_item else fl_item
			tmpsum += fl_item
	return tmpmax,tmpsum/len(featlist)

#3.normal num feature.
def feat_num(featlist):
	feat_vect = []
	max_feat,mean_feat = max_mean_feat(featlist)
	for item_feat in featlist:
		if item_feat == 'NA':
			tmpval = mean_feat / (max_feat+1.0)
		else:
			tmpval = float(item_feat) / (max_feat+1.0)
		feat_vect.append(tmpval)
	return np.array(feat_vect)
