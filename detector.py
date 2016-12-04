import os
import sys
import cv2
import numpy as np
import svm
import imageProcessor

class Detector:
	def __init__(self, winSize):
		self.classifier_ = None
		self.winSize_ = winSize
		self.descriptor_ = imageProcessor.getHOGDescriptor(winSize)

	def setClassifier(self, classifier):
		self.classifier_ = classifier

	# dataSet is an image-data set
	# labels is a list (not np.array)
	def train(self, dataSet, labels):
		featureVectorSet = []
		for img in dataSet:
			feature = self.__convert2RowSample(self.descriptor_(img), "float32")
			featureVectorSet.append(feature)

		featureVectorSet = np.array(featureVectorSet, dtype="float32")
		labels = np.array(labels, dtype="int32")

		self.classifier_.train(featureVectorSet, labels)

	def detect(self, img):
		if img is None:
			print "detector detect(): no image input"
			quit()
		
		featureVector = self.descriptor_(img)
		rowData = self.__convert2RowSample(featureVector, "float32")
		rowData = np.array([rowData], dtype="float32")

		res = self.classifier_.predict(rowData)
		return res

	def __convert2RowSample(self, vector, type):
		rowData = []
		for row in vector:
			rowData.append(row[0])
		rowData = np.array(rowData, dtype=type)
		return rowData

	def foo(self):
		img = cv2.imread("carSample.bmp")
		feature = self.descriptor_(img)
		print feature
		print len(feature)


if __name__ == '__main__':
	detector = Detector((128, 96))
	detector.foo()
