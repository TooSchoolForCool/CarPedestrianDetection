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

	# training the classifier in this detector
	# parameter:
	# 		dataSet: this is a set of image file.
	#		labels: a list of labels (NOT a np.array) which point out
	#			that each image in the dataSet belong to which class, correspondingly.
	def train(self, dataSet, labels):
		featureVectorSet = []
		for img in dataSet:
			# check the size of the training sample, if not equal to the windowSize then resize it
			if img.shape[0] != self.winSize_[1] or img.shape[1] != self.winSize_[0]:
				img = imageProcessor.resize(img, self.winSize_[0], self.winSize_[1])

			# get the feature from this img file, the output is an 2d np.array
			feature = self.descriptor_(img)
			# convert the feature into ROW_SAMPLE form for the training of classifier
			# the output is a 1d np.array
			feature = self.__convert2RowSample(feature, "float32")
			featureVectorSet.append(feature)

		# convert the vectorSet from a list of 1d np.array to a 2d np.array
		featureVectorSet = np.array(featureVectorSet, dtype="float32")
		# convert the list of labels to a 1d np.array
		labels = np.array(labels, dtype="int32")

		self.classifier_.train(featureVectorSet, labels)

	# 
	def predict(self, img):
		if img is None:
			print "detector detect(): no image input"
			quit()

		# check the size of the training sample, if not equal to the windowSize then resize it
		if img.shape[0] != self.winSize_[1] or img.shape[1] != self.winSize_[0]:
			img = imageProcessor.resize(img, self.winSize_[0], self.winSize_[1])

		# get the feature from this img file, the output is an 2d np.array
		feature = self.descriptor_(img)
		# convert the feature into ROW_SAMPLE form for the training of classifier
		# the output is a 1d np.array
		rowData = self.__convert2RowSample(feature, "float32")
		# convert the 1d np.array to a 2d np.array with only 1 row
		rowData = np.array([rowData], dtype="float32")

		# the return value is a 2d np.array, res[i][0] is the result for the i-th item
		res = self.classifier_.predict(rowData)
		return res[0][0]

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
