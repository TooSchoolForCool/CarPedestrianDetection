import os
import sys
import imutils
import cv2
import numpy as np

def HOG(img):
	winSize = (128, 96)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4. 
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	winStride = (8,8)
	padding = (8,8)
	locations = [[0,0]]

	hist = hog.compute(img, winStride, padding, locations)

	return hist

def getMultiScaleDetector(winSize):
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4. 
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	
	detector = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	return detector

def getHOGDescriptor(winSize):
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 9
	derivAperture = 1
	winSigma = 4. 
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 1
	nlevels = 64
	
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	return hog.compute


# loadImages images from the root directory and resize the image 
# in width and height respectively.
# parameters:
#		root: the root directory where the function look for img file
#		width: load the image in width of width(param)
#		height: load the image in height of height(param)
# return:
# 		images: a list of images loaded 		
def loadImages(root, width=None, height=None):
	if root is None:
		print "loadImages: path does not exist"
		quit(1)

	images = []

	for path, dirs, files in os.walk(root):
		for file in files:
			try:
				fileName, extension = file.split(".")
				if extension not in ["jpg", "bmp", "BMP", "ppm"]:
					continue
			except:
				continue

			imgPath = path + "/" + file
			img = cv2.imread(imgPath)

			if img is None:
				print "Cannot open file:", imgPath
				quit(1)

			if width is not None and height is not None:
				img = resize(img, width, height)

			# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			
			images.append(img)

	return images

# resize the img file
# parameters:
#		img: the img to be resized
#		width: new width
#		height: new height
# return:
# 		newImg: new image with width of new width and height of new height	
def resize(img, width, height):
	try:
		newImg = cv2.resize(img, None, fx = 1.0*width/img.shape[1], fy = 1.0*height/img.shape[0], interpolation = cv2.INTER_CUBIC)
	except:
		print width, height, img.shape[1], img.shape[0]
	return newImg

def tester():
	print "Tester for imageProcessor:"
	foo()

if __name__ == '__main__':
	tester()