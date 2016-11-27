import os
import cv2
import numpy as np
import imutils

from imutils.object_detection import non_max_suppression

CV_HOG_DEFAULT_PEOPLE_SVM = 0

class humanDetector:
	def __init__(self):
		self.classifierType_ = None
		self.classifier_ = None
		self.detectFuncPointer = [self.__detectHuman4cvDefaultHogSVM]

	def setDefaultSVM4Human(self):
		global CV_HOG_DEFAULT_PEOPLE_SVM

		self.classifierType_ = CV_HOG_DEFAULT_PEOPLE_SVM

		self.classifier_ = cv2.HOGDescriptor()
		self.classifier_.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


	def detectHuman(self, img):
		return self.detectFuncPointer[self.classifierType_](img)

	def __detectHuman4cvDefaultHogSVM(self, img):
		img = imutils.resize(img, height=max(128, img.shape[0]))
		img = imutils.resize(img, width=max(120, img.shape[1]))
		img = imutils.resize(img, width=min(720, img.shape[1]))

		rois, weights = self.classifier_.detectMultiScale(img, winStride=(4, 4),
			padding=(8, 8), scale=1.05)

		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rois = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rois])
		rois = non_max_suppression(rois, probs=None, overlapThresh=0.65)

		for (x1, y1, x2, y2) in rois:
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

		return img, rois

# Tester for human detector
def tester(root):
	if root == "":
		return

	detector = humanDetector()
	detector.setDefaultSVM4Human()

	numOfHuman, numOfNonHuman = 0, 0

	for path, dirs, files in os.walk(root):
		for file in files:
			imgPath = path + "/" + file
			img = cv2.imread(imgPath)

			if img is None:
				print "Cannot open file:", imgPath
				quit(1)

			img, rois = detector.detectHuman(img)

			if rois == []:
				numOfNonHuman += 1
			else:
				numOfHuman += 1

	print "number of human:", numOfHuman
	print "number of non-human", numOfNonHuman
	print "ratio of human:", 1.0 * numOfHuman / (numOfHuman + numOfNonHuman)



if __name__ == '__main__':
	tester("")

