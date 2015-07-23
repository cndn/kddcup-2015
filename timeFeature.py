import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import numpy as np
import time
def timePerActivity():
#-------------------construct timePerActivity.csv--------------
	start = time.clock()
	log = pd.read_csv("log_test.csv")
	enr = pd.read_csv("enrollment_test.csv")

	middle = time.clock() - start
	print "it takes " + str(middle) +" to load data"
	df = pd.DataFrame(np.zeros((121000,15)),\
		columns = ['enrollment_id','access_browser_time','access_server_time','nagivate_browser_time','nagivate_server_time',\
		'problem_browser_time','problem_server_time','page_close_server_time','page_close_browser_time','video_server_time',\
		'video_browser_time','discussion_server_time','discussion_browser_time','wiki_server_time','wiki_browser_time'])
	df["enrollment_id"] = enr["enrollment_id"]
	df_index = 0
	last = 1
	df = df.to_dict()
	for i in xrange((log.shape[0]-1)): #heihei
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
		if log["enrollment_id"][i+1] == log["enrollment_id"][i] and (parse(log["time"][i+1]) - parse(log["time"][i])).total_seconds() < 7200:
			timeSpan = (parse(log["time"][i+1]) - parse(log["time"][i])).total_seconds()
		else:
			timeSpan = 0
		df[event+"_"+source+"_time"][df_index] += timeSpan
	df = pd.DataFrame.from_dict(df)
	df.to_csv("timePerActivity_test.csv")

def timePerActivityLargeSpan(inputChoice,output):
#-------------------construct timePerActivityLargeSpan.csv--------------
	start = time.clock()
	log = pd.read_csv("log_"+inputChoice+".csv")
	enr = pd.read_csv("enrollment_"+inputChoice+".csv")

	middle = time.clock() - start
	print "it takes " + str(middle) +" to load data"
	df = pd.DataFrame(np.zeros((enr.shape[0],15)),\
		columns = ['enrollment_id','access_browser_time','access_server_time','nagivate_browser_time','nagivate_server_time',\
		'problem_browser_time','problem_server_time','page_close_server_time','page_close_browser_time','video_server_time',\
		'video_browser_time','discussion_server_time','discussion_browser_time','wiki_server_time','wiki_browser_time'])
	df["enrollment_id"] = enr["enrollment_id"]
	df_index = 0
	last = 2
	df = df.to_dict()
	for i in xrange(10):#log.shape[0]):
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
		if log["enrollment_id"][i+1] == log["enrollment_id"][i] and \
		(parse(log["time"][i+1]) - parse(log["time"][i])).total_seconds() < 40:
			df[event+"_"+source+"_time"][df_index] += 1
	df = pd.DataFrame.from_dict(df)
	df.to_csv(output)

if __name__ == '__main__':
	timePerActivityLargeSpan("train","timeSpanLessThan40_train.csv")