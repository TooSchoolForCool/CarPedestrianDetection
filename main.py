import os
import sys
import getopt
import cv2
import numpy as np
import svm
import detector
import imageProcessor

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

# My Test Case
MY_TEST_CASE = ["./dataSet/myTestCase"]

# All database sets
ALL_DATABASE = MIT_DATABASE + IN_CLASS_DATABASE

# test base
CAR_DATABASE = "/Users/zeyuzhang/Documents/HNU/Pattern Recognition/Project2/dataSet/classTest/car"
PEOPLE_DATABASE = "/Users/zeyuzhang/Documents/HNU/Pattern Recognition/Project2/dataSet/classTest/people"
IN_CLASS_TEST = [CAR_DATABASE, PEOPLE_DATABASE]

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

	# positive samples
	posSamples = imageProcessor.loadImages(IN_CLASS_HUMAN_SET_TRAIN)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	posSamples = imageProcessor.loadImages(MIT_HUMAN_SET)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	# negative samples
	negSamples = imageProcessor.loadImages(IN_CLASS_BACKGROUND_SET)
	dataSet += negSamples
	labels += [0 for i in range(len(negSamples))]

	# negSamples = imageProcessor.loadImages(IN_CLASS_CAR_SET_TEST)
	# dataSet += negSamples
	# labels += [0 for i in range(len(negSamples))]

	detector4human.train(dataSet, labels)

	return detector4human


def getCarDetector(winSize):
	svm4car = svm.SVM()
	svm4car.setLinearSVM()

	carDetector = detector.Detector(winSize)
	carDetector.setClassifier(svm4car)

	dataSet = []
	labels = []

	# positive samples
	posSamples = imageProcessor.loadImages(IN_CLASS_CAR_SET_TEST)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	posSamples = imageProcessor.loadImages(MIT_CAR_SET)
	dataSet += posSamples
	labels += [1 for i in range(len(posSamples))]

	# negative samples
	negSamples = imageProcessor.loadImages(IN_CLASS_BACKGROUND_SET)
	dataSet += negSamples
	labels += [0 for i in range(len(negSamples))]

	# negSamples = imageProcessor.loadImages(IN_CLASS_HUMAN_SET_TRAIN)
	# dataSet += negSamples
	# labels += [0 for i in range(len(negSamples))]

	# negSamples = imageProcessor.loadImages(IN_CLASS_HUMAN_SET_TEST)
	# dataSet += negSamples
	# labels += [0 for i in range(len(negSamples))]

	# negSamples = imageProcessor.loadImages(MIT_HUMAN_SET)
	# dataSet += negSamples
	# labels += [0 for i in range(len(negSamples))]

	carDetector.train(dataSet, labels)

	return carDetector

# tester for human
# dataBase: the root directory of the dataBase for testing
def tester4human(database):
	detector4human = getHumanDetector((64, 96))

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
	# humanDetector = getCarDetector((128, 96))
	humanDetector = getHumanDetector((64, 96))

	img = cv2.imread("3.jpg")
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img, rois = humanDetector.detectMultiScale(img, (20, 20), (8, 8), 1.2)
	# img, rois = humanDetector.detect(img)
	# img, rois = humanDetector.demo(img, (10, 10), 1.2)

	numOfHuman, numOfNonHuman = 0, 0
	for img in rois:
		res = humanDetector.predict(img)
		if res == 1.0:
			# path = "./test/" + str(numOfHuman) + ".jpg"
			# cv2.imwrite(path, img)
			numOfHuman += 1
		else:
			numOfNonHuman += 1
		path = "./test/" + str(numOfHuman+numOfNonHuman) + ".jpg"
		cv2.imwrite(path, img)

	print "Testing Human Detector"
	print "number of human:", numOfHuman
	print "number of non-human:", numOfNonHuman
	print "ratio of human:", 1.0 * numOfHuman / (numOfHuman + numOfNonHuman)
	print "-----------------------"

	database = ["./test"]

	for repo in database:
		images = imageProcessor.loadImages(repo)

		numOfHuman, numOfNonHuman = 0, 0
		for img in images:
			res = humanDetector.predict(img)
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

	# cv2.imshow("img", img)
	# cv2.waitKey(0)

def foo1():
	humanDetector = getHumanDetector((128, 96))

	img = cv2.imread("7.jpg")

	img, rois = humanDetector.demo(img, (10, 10), 1.2)

def demo():
	humanDetector = getCarDetector((128, 96))
	# humanDetector = getHumanDetector((64, 128))

	img = cv2.imread("5.jpg")
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img, rois = humanDetector.detect(img)
	# img, rois = humanDetector.detect(img)
	# img, rois = humanDetector.demo(img, (10, 10), 1.2)

	cv2.imshow("img", img)
	cv2.waitKey(0)

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
			elif arg in ["my", "MY", "My"]:
				dataBase = MY_TEST_CASE
			elif arg in ["all", "ALl"]:
				dataBase = ALL_DATABASE

	if isTestMode:
		tester4human(IN_CLASS_TEST)
		tester4car(IN_CLASS_TEST)
	else:
		foo1()


if __name__ == '__main__':
	main(sys.argv)