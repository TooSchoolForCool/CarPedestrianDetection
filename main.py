import os
import sys
import getopt
import cv2
import numpy as np
import svm
import imageProcessor
import humanDetector

"""
Global variable
"""

# in class test sets
IN_CLASS_HUMAN_SET_TRAIN = "./dataSet/inClassData/pedestrain/train"
IN_CLASS_HUMAN_SET_TEST = "./dataSet/inClassData/pedestrain/test"
IN_CLASS_CAR_SET_TRAIN = "./dataSet/inClassData/car/train"
IN_CLASS_CAR_SET_TEST = "./dataSet/inClassData/car/test"
IN_CLASS_BACKGROUND_SET = "./dataSet/inClassData/background"
IN_CLASS_DATABASE = [IN_CLASS_HUMAN_SET_TRAIN, IN_CLASS_HUMAN_SET_TEST, 
	IN_CLASS_CAR_SET_TRAIN, IN_CLASS_CAR_SET_TEST, IN_CLASS_BACKGROUND_SET]

# MIT DataBase
MIT_HUMAN_SET = "./dataSet/MITDatabase/pedestrians"
MIT_CAR_SET = "./dataSet/MITDatabase/cars"
MIT_DATABASE = [MIT_HUMAN_SET, MIT_CAR_SET]

# All database sets
ALL_DATABASE = MIT_DATABASE + IN_CLASS_DATABASE

"""
Global Variable Define ENDm
"""

# tester for human
# dataBase: the root directory of the dataBase for testing
def tester4human(dataBase):
	if dataBase is None:
		print "In main.py tester4human(): dataBase is None"
		quit(1)

	for repo in dataBase:
		print "Testing Human Detector"
		print "directory:", repo

		humanDetector.tester(repo)

		print "-----------------------"

# foo function, fool function :)
def foo():
	detector = svm.SVM()
	detector.setLinearSVM()

def foo1():
	train_pts = 30  

	rand1 = np.ones((train_pts,2)) * (-2) + np.random.rand(train_pts, 2)
	# print('rand1: ')
	# print(rand1)  

	rand2 = np.ones((train_pts,2)) + np.random.rand(train_pts, 2)  
	# print('rand2: ')  
	# print(rand2)  


	train_data = np.vstack((rand1, rand2))  
	train_data = np.array(train_data, dtype='float32')  
	train_label = np.vstack( (np.zeros((train_pts,1), dtype='int32'), np.ones((train_pts,1), dtype='int32')))


	svm = cv2.ml.SVM_create()  
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_LINEAR)  
	svm.setC(1.0)  

	ret = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label) 

	pt = np.array(np.random.rand(20,2) * 4 - 2, dtype='float32')  
	(ret, res) = svm.predict(pt)

	print("res = ")  
	print(res)  

# main function, parses the arguments from the command line
def main(argv):
	dataBase = IN_CLASS_DATABASE
	tester = tester4human
	isTestMode = True

	try:
		opts, args = getopt.getopt(argv[1:], 'r', ["test=", "realtime", "database="])
	except getopt.GetoptError, err:
		print str(err)
		quit(1)

	for opt, arg in opts:
		if opt in ["--test"]:
			isTestMode = True
			if arg in ["human", "Human"]:
				tester = tester4human
		elif opt in ["-r", "--realTime"]:
			isTestMode = False
		elif opt in ["--database"]:
			if arg in ["class", "Class"]:
				dataBase = IN_CLASS_DATABASE
			elif arg in ["mit", "MIT"]:
				dataBase = MIT_DATABASE
			elif arg in ["all", "ALl"]:
				dataBase = ALL_DATABASE

	if isTestMode:
		tester(dataBase)
	else:
		foo()


if __name__ == '__main__':
	main(sys.argv)