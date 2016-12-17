import os
import cv2
import numpy as np
import imutils
import imageProcessor

from imutils.object_detection import non_max_suppression

# svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                     svm_type = cv2.SVM_C_SVC,
#                     C=2.67, gamma=5.383 )

class SVM:
	def __init__(self):
		# the pointer point to the classifier
		self.svm_ = None

	# detect car object in the image
	# parameter:
	# 		img: the image your want to detect
	# return:
	#		img: the modified image file with detected object rectangled
	#		rois: regions of interst, i.e., the rectangle area (x1, y1, x2, y2)
	def predict(self, vector):
		ret, res = self.svm_.predict(vector)
		return res

	def setLinearSVM(self):
		svm = cv2.ml.SVM_create()  
		svm.setType(cv2.ml.SVM_C_SVC)
		svm.setKernel(cv2.ml.SVM_LINEAR) 
		svm.setC(0.5)
		self.svm_ = svm

	def train(self, dataSet, labels):
		self.svm_.train(dataSet, cv2.ml.ROW_SAMPLE, labels)

	def getSupportVectors(self):
		self.svm_.save('svm_data.dat')
		return self.svm_.getSupportVectors()


# Tester for human detector
# root: the root directory of the testing files
def tester(root):
	if root is None:
		print "carDetector-tester(): img is None"

	detector = humanDetector()
	detector.setDefaultSVM4Human()

	images = imageProcessor.loadImages(root)
	if images == []:
		print "carDetector-tester(): Cannot find any image in path:", root
		return 

	numOfCars, numOfNonCars = 0, 0
	for img in images:
		img, rois = detector.detectHuman(img)

		if rois == []:
			numOfNonCars += 1
		else:
			numOfCars += 1

	print "number of cars:", numOfCars
	print "number of non-cars", numOfNonCars
	print "ratio of car:", 1.0 * numOfCars / (numOfCars + numOfNonCars)


if __name__ == '__main__':
	tester("")

