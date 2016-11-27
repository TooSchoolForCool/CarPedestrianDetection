import os
import sys
import cv2
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
IN_CLASS_SETS = [IN_CLASS_HUMAN_SET_TRAIN, IN_CLASS_HUMAN_SET_TEST, 
	IN_CLASS_CAR_SET_TRAIN, IN_CLASS_CAR_SET_TEST, IN_CLASS_BACKGROUND_SET]

"""
Global Variable Define END
"""

def tester4human():
	global IN_CLASS_SETS

	for repo in IN_CLASS_SETS:
		print "Testing Human Detector"
		print "directory:", repo

		humanDetector.tester(repo)

		print "-----------------------"

def foo():
	img = cv2.imread("1.ppm")
	cv2.imshow("img", img)
	cv2.waitKey(0)

def main(argv):
	# tester4human()
	foo()


if __name__ == '__main__':
	main(sys.argv)