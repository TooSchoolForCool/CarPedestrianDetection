import os
import sys
import imageProcessor

def main(argv):
	img = cv2.imread("1.ppm")
	cv2.imshow("image", img)
	cv2.waitKey(0)

if __name__ == '__main__':
	main(sys.argv)