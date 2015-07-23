import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
def trainData():
	totalTrainSet = pd.read_csv("aggregationFeature.csv")
	totalLabelSet = pd.read_csv("truth_train.csv")
	totalTestSet = pd.read_csv("aggregationFeature_test.csv")
	enr_train = pd.read_csv("enrollment_train.csv")
	enr_test = pd.read_csv("enrollment_test.csv")

	# x = totalTrainSet.as_matrix()[:,1:]
	# submission = totalTestSet.as_matrix()[:,1:]
	x = np.log(totalTrainSet.as_matrix()[:,1:] + 1)
	submission = np.log(totalTestSet.as_matrix()[:,1:] + 1)

	total = np.concatenate((x,submission))
	scaler = preprocessing.StandardScaler().fit(total)
	x = scaler.transform(x)
	y = totalLabelSet.as_matrix()[:,1:]
	submission = scaler.transform(submission)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	X_train, X_CV, y_train, y_CV = train_test_split(X_train, y_train, test_size=0.2)

	dtotal = xgb.DMatrix(x)
	dtrain = xgb.DMatrix(X_train,label = y_train)
	dCV = xgb.DMatrix(X_CV,label = y_CV)
	dtest = xgb.DMatrix(X_test)
	dsubmission = xgb.DMatrix(submission)

	dtrain.save_binary('train.buffer') # make loading faster in next time

	param = {'bst:max_depth':4, 'bst:eta':.3, 'silent':1, 'objective':'binary:logistic' }
	param['nthread'] = 4
	plst = param.items()

	evallist  = [(dCV,'eval'), (dtrain,'train')]

	num_round = 30
	bst = xgb.train( plst, dtrain, num_round, evallist )

	# dump model
	bst.dump_model('dump.raw.txt')
	# # dump model with feature map
	# bst.dump_model('dump.raw.txt','featmap.txt')

	ypred = bst.predict(dtest)
	truth = []
	for item in y_test:
		truth.append(item[0])
	print roc_auc_score(truth,ypred)
	# #submission
	# pre = bst.predict(dtotal)
	# df = pd.DataFrame({"enrollment_id":enr_train.enrollment_id,"predict":pre})
	# df.to_csv("xgblogscale_2.csv")

	pre = bst.predict(dsubmission)
	df = pd.DataFrame({"enrollment_id":enr_test.enrollment_id,"predict":pre})
	df.to_csv("612FB.csv")

if __name__ == '__main__':
	trainData()