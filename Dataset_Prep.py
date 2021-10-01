# usage - python Dataset_Prep.py --image path_to_the_dataset
# import the necessary packages
import argparse
import cv2
from imutils import paths
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
for imagePath in paths.list_images(args["image"]):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	#cv2.imshow("Image", image)

	# apply basic thresholding -- the first parameter is the image
	# we want to threshold, the second value is is our threshold
	# check; if a pixel value is greater than our threshold (in this
	# case, 200), we set it to be BLACK, otherwise it is WHITE.
	(T, threshInv) = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
	#cv2.imshow("Threshold Binary Inverse", threshInv)

	cnts = cv2.findContours(threshInv.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	crop = threshInv[y:y+h,x:x+w]
	crop_resized = cv2.resize(image, (100, 100))
	cv2.imwrite(imagePath,crop)
	# cv2.imwrite(imagePath,crop_resized)
	print (imagePath)