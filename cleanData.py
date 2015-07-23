import pandas as pd
import numpy as np
import time

def coursesEveryUser():
#------------------ construct coursesEveryUser.csv--------------
	log = pd.read_csv("log_train.csv")
	enr = pd.read_csv("enrollment_train.csv")
	username = []
	num_course = []
	for i in xrange(enr.shape[0]):
		if i%1000 == 0:
			print i
		if enr.username[i] not in username:
			username.append(enr.username[i])
			num_course.append(enr[enr.username == enr.username[i]].shape[0])
	df = pd.DataFrame({'username':username,'num_course':num_course})
	df.to_csv("coursesEveryUser.csv")

def usersEveryCourse():
#------------------ construct usersEveryCourse.csv--------------
	log = pd.read_csv("log_train.csv")
	enr = pd.read_csv("enrollment_train.csv")
	course_id = []
	num_user = []
	for i in xrange(enr.shape[0]):
		if i%1000 == 0:
			print i
		if enr.course_id[i] not in course_id:
			course_id.append(enr.course_id[i])
			num_user.append(enr[enr.course_id == enr.course_id[i]].shape[0])
	df = pd.DataFrame({'course_id':course_id,'num_user':num_user})
	df.to_csv("usersEveryCourse.csv")

def eventEveryEnrollment():
#------------------ construct eventEveryEnrollment.csv--------------
#-----numexpr is needed----------------------
#-----nagivate-------------------------------
#-----testset cuowei dont know why
	start = time.clock()
	log = pd.read_csv("log_test.csv")
	enr = pd.read_csv("enrollment_test.csv")

	# nagivate_browser = ("nagivate","browser")
	# nagivate_server = ("nagivate","server")
	# access_browser = ("access","browser")
	# access_server = ("access","server")
	# problem_browser = ("problem","browser")
	# problem_server = ("problem","server")
	# page_close_browser = ("page_close","browser")
	# page_close_server = ("page_close","server")
	# video_browser = ("video","browser")
	# video_server = ("video","server")
	# discussion_browser = ("discussion","browser")
	# discussion_server = ("discussion","server")
	# wiki_browser = ("wiki","browser")
	# wiki_server = ("wiki","server")
	# ESList = [access_browser,access_server,nagivate_browser,nagivate_server,\
	# 	problem_browser,problem_server,page_close_server,page_close_browser,video_server,\
	# 	video_browser,discussion_server,discussion_browser,wiki_server,wiki_browser]
	middle = time.clock() - start
	print "it takes " + str(middle) +" to load data"
	df = pd.DataFrame(np.zeros((121000,15)),\
		columns = ['enrollment_id','access_browser','access_server','nagivate_browser','nagivate_server',\
		'problem_browser','problem_server','page_close_server','page_close_browser','video_server',\
		'video_browser','discussion_server','discussion_browser','wiki_server','wiki_browser'])
	df["enrollment_id"] = enr["enrollment_id"]
	df_index = 0
	last = 1
	df = df.to_dict()
	for i in xrange(log.shape[0]):
		if i%10000==0:
			print i
			print df_index,last
			print "timecost: " + str(time.clock()-start)
		enrollment_id = log["enrollment_id"][i]
		if enrollment_id != last:
		 	df_index += 1
		 	last = enrollment_id 
		 	
		event = log["event"][i]
		source = log["source"][i]

		df[event+"_"+source][df_index] += 1		
	df = pd.DataFrame.from_dict(df)
	df.to_csv("eventEveryEnrollment_test.csv")


def coursesEveryEnrollment():
#------------------ construct coursesEveryEnrollment.csv--------------
	enr = pd.read_csv("enrollment_test.csv")
	coursesEveryUser = pd.read_csv("coursesEveryUser_test.csv")
	# df = pd.DataFrame(np.zeros((121000,3)),columns = ['enrollment_id','username','num_course'])
	# df["enrollment_id"] = enr["enrollment_id"]
	# df["username"] = enr["username"]
	# for i in xrange(df.shape[0]):
	# 	df.loc["num_course",i] = coursesEveryUser[coursesEveryUser.username == df.username[i]].as_matrix()[0][1]
 #        print i
 	df = enr
 	df = df.merge(coursesEveryUser, on = 'username',how='left')
	print df.head(50).to_string()
	df.to_csv("coursesEveryEnrollment_test.csv")

def localTest():
	train = pd.read_csv("enrollment_localtrain.csv")
	test = pd.read_csv("enrollment_localtest.csv")
	enrid_train = train.enrollment_id.as_matrix()
	print enrid_train
	enrid_test = test.enrollment_id.as_matrix()
	feature = pd.read_csv("feature.csv")
	feature = feature.set_index("enrollment_id")
	print feature.to_string()
	feature_localtrain = feature.loc[enrid_train]
	feature_localtest = feature.loc[enrid_test]
	print 'writing csv'
	feature_localtrain.to_csv("feature_localtrain.csv")
	feature_localtest.to_csv("feature_localtest.csv")
def numOfLog():
	log = pd.read_csv("log_train.csv")
	enr = pd.read_csv("enrollment_train.csv")
	df = pd.DataFrame(np.zeros((log.shape[0],2)),columns = ['enrollment_id','num_log'])
	df["enrollment_id"] = enr["enrollment_id"]
	df = df.to_dict()
	for i in xrange(enr.shape[0]):
		df["num_log"][i] = log[log.enrollment_id == i].shape[0]
		if i%1000 == 0:
			print i
	df.to_csv("numOfLog")

def splitByCourse():
	course = pd.read_csv("allcourse.csv")
	course_id = course.course_id
	course_index = course.index
	dictionary = dict(zip(course.course_id,course.index))
	featureToSplit = pd.read_csv("featureToSp.csv")
	
		
	for item in course_id:
		df = featureToSplit[featureToSplit.course_id == item]
		truth = df[['enrollment_id','dropCourse']]
		truth.to_csv(str(dictionary[item]) + "_truth.csv",index = False)
		df = df.drop('course_id',1)
		df = df.drop('dropCourse',1)
		df.to_csv(str(dictionary[item])+'.csv',index = False)

		
		# dfList[dictionary[featureToSplit.course_id[i]]].append(featureToSplit[i])
	

if __name__ == '__main__':
	# coursesEveryUser()
	# usersEveryCourse()
    # eventEveryEnrollment()
	# coursesEveryEnrollment()
	# localTest()
	# numOfLog()
	# print df.head(200).to_string()
	# log1 = log[log.enrollment_id == 1]
 #    nameList = []
 	# splitByCourse()
	# # print log1.to_string()
	# log = pd.read_csv("log_train.csv")
	# enr = pd.read_csv("enrollment_train.csv")
	# print log[(log.event == 'nagivate') & (log.source == 'server')]
	# print log.query("event == 'nagivate' and source == 'server'")
