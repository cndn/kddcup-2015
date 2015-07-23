import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
def trainAndTest():
	#-------------load train, label and test set-----------------
	totalTrainSet = pd.read_csv("45features.csv")
	totalLabelSet = pd.read_csv("truth_train.csv")
	totalTestSet = pd.read_csv("45features_test.csv")
	enr_train = pd.read_csv("enrollment_train.csv")
	enr_test = pd.read_csv("enrollment_test.csv")
    #------------- standard scale for train and test set altogether -----------------
	x = totalTrainSet.as_matrix()[:,1:]
	X_test = totalTestSet.as_matrix()[:,1:] #for submission
	submission = totalTestSet.as_matrix()[:,1:] #when dump scaler add 1:

	# x = np.log(totalTrainSet.as_matrix()[:,1:] + 1)
	# X_test = np.log(totalTestSet.as_matrix()[:,1:] + 1) #for submission
	# submission = np.log(totalTestSet.as_matrix()[:,1:] + 1)

	# print x[:,0]
	y = totalLabelSet.as_matrix()[:,1:]
	
	total = np.concatenate((x,X_test))
	scaler = preprocessing.StandardScaler().fit(total)
	x = scaler.transform(x)
	x[:,0] = total[:len(x),0]
	
	joblib.dump(scaler,'scaler.pkl')    # dump the scaler

    #------------- PCA for train and test set altogether (seem useless) -----------------
	# pca = PCA(n_components = 4)
	# pca.fit_transform(total)
	# x = pca.transform(x)
	# print (pca.explained_variance_ratio_)
	# print sum(pca.explained_variance_ratio_)
	# joblib.dump(pca, 'pca_total_4.pkl')


	#------------- Initialize models -----------------
	start = time.clock()
	print "start training"
	names = ["KNN","Decision Tree","LinearSVM","SVC","Random Forest", "AdaBoost", "Naive Bayes"]
	classifiers = [	KNeighborsClassifier(256),\
	DecisionTreeClassifier(max_depth=5),\
	SVC(kernel="linear", C=025),\
	SVC(gamma=1, C=5,probability = True),\
		RandomForestClassifier(max_depth= 5, n_estimators=30, max_features=10, n_jobs = -1),\
		AdaBoostClassifier(),\
		GaussianNB()]

	#------------- split the training set randomly-----------------
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	for name, clf in zip(names, classifiers):
	#------------- fit and dump-----------------
		if name == "AdaBoost":                 # name in [] - fit every model in the list
			clf.fit(X_train[:,:], y_train)#,sample_weight =  np.array(map(lambda x: abs(x-0.5) ,y_train)).transpose()[0]) # sample weight >0.5 stress negative 
			#joblib.dump(clf, 'classifier_'+ name + '.pkl')  #generate classifier.pkl
	#------------- score and F1 score -----------------------------
			score = clf.score(X_test[:,:],y_test)
			pre = clf.predict(X_train[:,:])
			f1_train = f1_score(y_train, pre, average=None)
			f1_test = f1_score(y_test,clf.predict(X_test[:,:]),average=None)
			# print "classifier:  "+ name + "   score:  "+ str(score)
			# print "classifier:  "+ name + "   F1 score on training set:  "+ str(f1_train)
			# print "classifier:  "+ name + "   F1 score on test set:  "+ str(f1_test)
			print "importance:"
			print clf.feature_importances_

	#------------- output probability and AUC score -----------------------------
			pre2class = clf.predict_proba(X_test[:,:])
			pre = []
			for predict in pre2class:
				if predict[1]>0.48:
					pre.append(predict[1])
				else:
					pre.append(predict[1])
			truth = []
			for item in y_test:
				truth.append(item[0])
			print 'AUC test score:'
			print roc_auc_score(truth,pre)
			# df = pd.DataFrame({"enrollment_id":X_test[:,0],"predict":pre,"truth":truth})
			# df.to_csv("see20.csv")

	#------------- 2nd feature -----------------------------
			# pre2class = clf.predict_proba(x[:,:])
			# predictTrain = []
			# for predict in pre2class:
			# 	predictTrain.append(predict[1])
			# df = pd.DataFrame({"enrollment_id":enr_train.enrollment_id,"predict":predictTrain})
			# df.to_csv("KNN_256.csv")

			# pre2class = clf.predict_proba(submission[:,:])
			# predictSub = []
			# for predict in pre2class:
			# 	predictSub.append(predict[1])
			# df = pd.DataFrame({"enrollment_id":enr_test.enrollment_id,"predict":predictSub})
			# df.to_csv("KNN_256_test.csv")
			# print 'finish'

	print "timecost: " + str(time.clock()-start)

def submission(): 
	#-------------load necessary csv-----------------
	totalTrainSet = pd.read_csv("xuemeitest.csv")
	enr = pd.read_csv("enrollment_test.csv")
	#-------------scale the test set-----------------
	# pkl of scaler should be moved to current directory
	x = totalTrainSet.as_matrix()[:,1:]
	scaler = joblib.load('scaler.pkl')
	x = scaler.transform(x)
	#-------------do PCA on the test set-----------------
	# pkl of PCA should be moved to current directory
	# pca = joblib.load('pca_total_4.pkl')
	# x = pca.transform(x)
	# print (pca.explained_variance_ratio_)
	# print sum(pca.explained_variance_ratio_)

	names = ["AdaBoost"]
	clfList = []
	for name in names:
	    clfList.append(joblib.load('classifier_'+ name + '.pkl'))
	for clf in clfList:
		pre2class = clf.predict_proba(x)
		pre = []
		for predict in pre2class:
			if predict[1]>0.48:
				pre.append(predict[1])  # If 1
			else:
				pre.append(predict[1])  # If 0, output 0-1 classification according to the threshold
		print pre
		df = pd.DataFrame({"enrollment_id":enr.enrollment_id,"predict":pre})
		df.to_csv(str(clf)[:3]+".csv")

def chooseParameter(parameterList):
	totalTrainSet = pd.read_csv("xuemei69.csv")
	totalLabelSet = pd.read_csv("truth_train.csv")
	totalTestSet = pd.read_csv("xuemei69test.csv")

	x = totalTrainSet.as_matrix()[:,:]
	y = totalLabelSet.as_matrix()[:,1:]
	X_test = totalTestSet.as_matrix()[:,:] #for submission
	total = np.concatenate((x,X_test))
	scaler = preprocessing.StandardScaler().fit(total)
	x = scaler.transform(x)
	x[:,0] = total[:len(x),0]
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	tmp = []
	for item in y_train:
		tmp.append(item[0])
	y_train = tmp
	for parameter in parameterList:
		aucList = []
		for iteration in range(1):
			clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,max_depth=parameter, random_state=0)
			clf.fit(X_train[:,1:], y_train)#,sample_weight =  np.array(map(lambda x: abs(x-0.5) ,y_train)).transpose()[0]) 
			# sample weight >0.5 stress negative
			score = clf.score(X_test[:,1:],y_test)
			pre2class = clf.predict_proba(X_test[:,1:])
			pre = []
			for predict in pre2class:
				if predict[1]>0.48:
					pre.append(predict[1])
				else:
					pre.append(predict[1])
			truth = []
			for item in y_test:
				truth.append(item[0])
			aucList.append(roc_auc_score(truth,pre))

		print (parameter,sum(aucList)/(iteration+1))


if __name__ == '__main__':
	trainAndTest()
	# submission()
	# chooseParameter([1,2,3,4,5])