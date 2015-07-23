# kddcup-2015
This is for "bigSisterIsWatchingYou" record of kdd cup 2015.

Unfortunately, the dataset seems not available on official site after competition ends.If anyone is interested in
the code and results, you'd better find the dataset elsewhere on the Internet.

Features
---------------------
There are 45 total features we have tested on.

1.Upper level interaction between courses and users
    1) dropRate - rate of being dropped in history for a course
    2) compeleteRate - rate of completing course in history for a user
    3) num_course - number of courses a user has enrolled
    4) learnedObject - number of objects a user has browsed

2.User behavior 
    5) access_browser - total number of "access" combined with "browser"
    6) access_server - total number of "access" combined with "server"
    7) discussion_server - total number of "discussion" combined with "server"
    8) navigate_server - total number of "navigate" combined with "server"
    9) page_close_browser - total number of "page_close" combined with "browser"
    10) problem_browser - total number of "problem" combined with "browser"
    11) problem_server - total number of "problem" combined with "server"
    12) video_browser - total number of "video" combined with "browser"
    13) wiki_server - total number of "wiki" combined with "server"

3.Time features
    14 - 22) X_X_time - total time cost of a certain behavior (time cost for behavior A is defined as the time span between A and its next operation, operation of more than 2 hours would be dropped)
    23 - 31) X_X_time_1800 - number of operations which last more than 1800 seconds (half an hour)
    32 - 40) X_X_time_40 - number of operations which last less than 40 seconds (maybe indication for frequent skip[1]) 
    
    Here what an operation "lasts" indicates may differ from different operations. For video, it means the
    watching lasts for some time; while for "page_close", the user may leave for some time before come back again.

    41 - 44) interval[i] (i = 1,2,3,4) - time span between recent participation and the last one of it, if less than 4 participation, set to 30( To be open to question)
    45 course_id - since there are not too many courses(20+)

Models & Tools
------------------------
Python package use:
    scikit-learn pandas xgboost (and all their dependencies like numpy, scipy)

We have tested on many models, where effective ones include: (parameters are chosen with cross validation)
    Adaboost
    Random Forest
    Xgboost

Running models both directly and after log scale of level 1 features above, we get aggregation features(csv in
repository), combining them to some of level 1 features we find important, and do the prediction based on these 2
level features using Neural Network(nntool in matlab). But this step does not bring us better performance, seems 1 
level features should be re-considered carefully in the future, and maybe more 1 level features should be explored.
    






[1]http://lytics.stanford.edu/datadriveneducation/papers/yangetal.pdf
                
