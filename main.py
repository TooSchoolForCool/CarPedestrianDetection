import os
import sys
import getopt
import cv2
import numpy as np
import svm
import detector
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
Global Variable Define END
"""



def getHumanDetector(winSize):
	svm4human = svm.SVM()
	svm4human.setLinearSVM()

	detector4human = detector.Detector(winSize)
	detector4human.setClassifier(svm4human)

	dataSet = []
	labels = []

	posSamples = imageProcessor.loadImages(IN_CLASS_HUMAN_SET_TRAIN)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	negSamples = imageProcessor.loadImages(IN_CLASS_BACKGROUND_SET)
	dataSet += negSamples
	labels += [0 for i in range(len(negSamples))]

	detector4human.train(dataSet, labels)

	return detector4human


def getCarDetector(winSize):
	svm4car = svm.SVM()
	svm4car.setLinearSVM()

	carDetector = detector.Detector(winSize)
	carDetector.setClassifier(svm4car)

	dataSet = []
	labels = []

	posSamples = imageProcessor.loadImages(IN_CLASS_CAR_SET_TRAIN)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	negSamples = imageProcessor.loadImages(IN_CLASS_BACKGROUND_SET)
	dataSet += negSamples
	labels += [0 for i in range(len(negSamples))]

	carDetector.train(dataSet, labels)

	return carDetector

# tester for human
# dataBase: the root directory of the dataBase for testing
def tester4human(database):
	detector4human = getHumanDetector((96, 168))

	for repo in database:
		images = imageProcessor.loadImages(repo)

		numOfHuman, numOfNonHuman = 0, 0
		for img in images:
			res = detector4human.predict(img)
			if res == 1.0:
				numOfHuman += 1
			else:
				numOfNonHuman += 1

		print "Testing Human Detector"
		print "directory:", repo
		print "number of human:", numOfHuman
		print "number of non-human:", numOfNonHuman
		print "ratio of human:", 1.0 * numOfHuman / (numOfHuman + numOfNonHuman)
		print "-----------------------"


# tester for human
# dataBase: the root directory of the dataBase for testing
def tester4car(database):
	carDetector = getCarDetector((128, 96))

	for repo in database:
		images = imageProcessor.loadImages(repo)

		numOfCars, numOfNonCars = 0, 0
		for img in images:
			res = carDetector.predict(img)
			if res == 1.0:
				numOfCars += 1
			else:
				numOfNonCars += 1

		print "Testing Car Detector"
		print "directory:", repo
		print "number of cars:", numOfCars
		print "number of non-cars:", numOfNonCars
		print "ratio of cars:", 1.0 * numOfCars / (numOfCars + numOfNonCars)
		print "-----------------------"

# foo function, fool function :)
def foo():
	img = cv2.imread("humanSample.jpg")
	print img.shape

def foo1():
	train_pts = 30  

	rand1 = np.ones((train_pts,2)) * (-2) + np.random.rand(train_pts, 2)
	# print('rand1: ')
	# print(rand1)  

	rand2 = np.ones((train_pts,2)) + np.random.rand(train_pts, 2)  
	# print('rand2: ')  
	# print(rand2)  


	train_data = np.vstack((rand1, rand2))
	# print type(train_data)
	# print train_data, "\n---------------------"
	train_data = np.array(train_data, dtype='float32')
	# print train_data
	# print type(train_data)


	train_label = [[1 if i < 30 else 0] for i in range(60)]
	train_label = np.array(train_label, dtype='int32')
	# print train_label


	svm = cv2.ml.SVM_create()  
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_LINEAR)  
	svm.setC(1.0)  

	print train_data.shape
	print train_label.shape

	ret = svm.train(train_data, cv2.ml.ROW_SAMPLE, train_label) 

	pt = np.array(np.random.rand(20,2) * 4 - 2, dtype='float32')  

	print pt.shape

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
			elif arg in ["car", "Car"]:
				tester = tester4car
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