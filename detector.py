import os
import sys
import cv2
import numpy as np
import imutils
import svm
import imageProcessor

from imutils.object_detection import non_max_suppression

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
			feature = self.featureExtract(img)
			# convert the feature into ROW_SAMPLE form for the training of classifier
			# the output is a 1d np.array
			feature = self.__convert2RowSample(feature, "float32")
			featureVectorSet.append(feature)

		# convert the vectorSet from a list of 1d np.array to a 2d np.array
		featureVectorSet = np.array(featureVectorSet, dtype="float32")
		# convert the list of labels to a 1d np.array
		labels = np.array(labels, dtype="int32")

		self.classifier_.train(featureVectorSet, labels)

	# predict the class of the input image
	# parameter:
	#		img: the input image file
	# return:
	#		res[0][0]: the result of the classification, res itself is a 2d np.array
	# 			res[i][0] means the i-th result
	def predict(self, img):
		if img is None:
			print "detector detect(): no image input"
			quit()

		# check the size of the training sample, if not equal to the windowSize then resize it
		if img.shape[0] != self.winSize_[1] or img.shape[1] != self.winSize_[0]:
			img = imageProcessor.resize(img, self.winSize_[0], self.winSize_[1])

		# get the feature from this img file, the output is an 2d np.array
		feature = self.featureExtract(img)
		# convert the feature into ROW_SAMPLE form for the training of classifier
		# the output is a 1d np.array
		rowData = self.__convert2RowSample(feature, "float32")
		# convert the 1d np.array to a 2d np.array with only 1 row
		rowData = np.array([rowData], dtype="float32")

		# the return value is a 2d np.array, res[i][0] is the result for the i-th item
		res = self.classifier_.predict(rowData)
		return res[0][0]

	def detectMultiScale(self, img, winStride, padding, scale):

		img = imutils.resize(img, width=min(960, img.shape[1]))

		multiScaledImgs = []
		while img.shape[1] >= self.winSize_[0] and img.shape[0] >= self.winSize_[1]:
			multiScaledImgs.append(img)
			img = imutils.resize(img, width=int(img.shape[1]/scale))

		rois = []
		scaler = 1

		i = 0
		for img in multiScaledImgs:
			print img.shape
			for curY in xrange(0, img.shape[0] - self.winSize_[1] + 1, winStride[1]):
				for curX in xrange(0, img.shape[1] - self.winSize_[0] + 1, winStride[0]):
					cropped = img[curY:curY+self.winSize_[1], curX:curX+self.winSize_[0]].copy()
					res = self.predict(cropped)

					if res == 1.0:
						# print "This is a human"
						# cv2.imshow("human", cropped)
						# cv2.waitKey(0)

						# path = "./dataSet/myTestCase/" + str(i) + ".jpg"
						# cv2.imwrite(path, cropped)
						# i += 1
						rois.append(cropped)

						# width = int(self.winSize_[0] * scaler)
						# height = int(self.winSize_[1] * scaler)
						# rois.append( (curX, curY, curX+width, curY+height) )
			scaler *= scale

		img = multiScaledImgs[0]

		# for (x1, y1, x2, y2) in rois:
		# 	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

		return img, rois


	def detect(self, img):

		if img.shape[0] < self.winSize_[1] or img.shape[1] < self.winSize_[0]:
				img = imageProcessor.resize(img, self.winSize_[0], self.winSize_[1])

		# detector = cv2.HOGDescriptor()
		# detector.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
		# detector.save("hog.xml")

		detector = imageProcessor.getMultiScaleDetector(self.winSize_)
		supportVector = self.classifier_.getSupportVectors()[0]
		supportVector = self.__convert2ColSample(supportVector, "float32")

		detector.setSVMDetector( supportVector )

		img = imutils.resize(img, width=min(960, img.shape[1]))

		pad = 15
		rois, weights = detector.detectMultiScale(img, winStride=(4, 4),
			padding=(pad, pad), scale=1.2)


		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rois = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rois])
		rois = non_max_suppression(rois, probs=None, overlapThresh=0.65)

		for (x1, y1, x2, y2) in rois:
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

		return img, rois

	def demo(self, img, winStride, scale):
		multiScaledImgs = []

		while img.shape[1] >= self.winSize_[0] and img.shape[0] >= self.winSize_[1]:
			multiScaledImgs.append(img)
			img = imutils.resize(img, width=int(img.shape[1]/scale))

		for img in multiScaledImgs:
			for curY in xrange(0, img.shape[0] - self.winSize_[1] + 1, winStride[1]):
				for curX in xrange(0, img.shape[1] - self.winSize_[0] + 1, winStride[0]):
					tmpImg = img.copy()

					cv2.rectangle(tmpImg, (curX, curY), (curX+self.winSize_[0], curY+self.winSize_[1]), (0, 0, 255), 2)
					cv2.imshow("demo", tmpImg)

					cv2.waitKey(1)

		return (1, 1)

	def featureExtract(self, img):
		feature = self.descriptor_(img)
		# augment vector
		# augment = np.array([1], dtype="float32")
		# augment = np.array([augment], dtype="float32")
		# feature = np.vstack((feature, augment))
		return feature

	def __convert2RowSample(self, vector, type):
		rowData = []
		for row in vector:
			rowData.append(row[0])
		rowData = np.array(rowData, dtype=type)
		return rowData

	def __convert2ColSample(self, vector, type):
		colVector = []
		for col in vector:
			colVector.append([col])
		colVector = np.array(colVector, dtype=type)
		return colVector

	def foo(self):
		img = cv2.imread("carSample.bmp")
		feature = self.descriptor_(img)
		print feature
		print len(feature)


if __name__ == '__main__':
	detector = Detector((128, 96))
	detector.foo()
