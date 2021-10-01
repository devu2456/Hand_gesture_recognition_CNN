from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from tool.gesture_recognition import MotionDetector
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import h5py
from imutils import paths
import imutils
from PIL import Image
import keyboard

classnames=['back','close','enter','maximize','minimize','next','previous','spacebar','volumedown','volumeup'] 
key_press = ['backspace','alt+f4','enter','windows+up','windows+down','right','left','space','Volume_Down','Volume_UP']
# load json and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#keyboard.press_and_release('shift+s, space')

camera = cv2.VideoCapture(0)
ROI = "10,350,225,590"

# unpack the hand ROI, then initialize the motion detector and the total number of
# frames read thus far
(top, right, bot, left) = np.int32(ROI.split(","))
md = MotionDetector()
numFrames = 0
consec = 0
previous = "none"
pred = "nil"
maximum_indices = []
maximum_indices.append("0")
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	frame = cv2.flip(frame, 1)
	# resize the frame and flip it so the frame is no longer a mirror view
	frame = imutils.resize(frame, width=600)
	#frame = cv2.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]

	# extract the ROI, passing in right:left since the image is mirrored, then
	# blur it slightly
	roi = frame[top:bot, right:left]
	hand = roi.copy()
	#gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# if we not reached 32 initial frames, then calibrate the skin detector
	if numFrames < 32:
		md.update(gray)
		

	# otherwise, detect skin in the ROI
	else:
		# detect motion (i.e., skin) in the image
		skin = md.detect(gray)


		# check to see if skin has been detected
		if skin is not None:
			# unpack the tuple and draw the contours on the image
			(thresh, c) = skin
			#masked = cv2.bitwise_and(hand, hand, mask=thresh)
			#gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
			#edged = imutils.auto_canny(gray)
			(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			c = max(cnts, key=cv2.contourArea)
			(x, y, w, h) = cv2.boundingRect(c)
			logo = thresh[y:y + h, x:x + w]
			logo = cv2.resize(logo, (100, 100))
			logo =  cv2.cvtColor(logo, cv2.COLOR_GRAY2RGB)
			im_pil = Image.fromarray(logo)
			img_pred = image.img_to_array(im_pil)
			img_pred = np.expand_dims(img_pred, axis = 0)
			rslt = loaded_model.predict(img_pred)
			A = np.array(rslt[0])
			maximum_indices = np.where(A==max(rslt[0]))
			pred = str(classnames[int(maximum_indices[0][0])])
			cv2.putText(clone, pred, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 2)
			cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv2.imshow("Thresh", thresh)
			

	# draw the hand ROI and increment the number of processed frames
	cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
	if (previous==pred):
		consec+=1
	
	previous=pred

	if (consec>50):
		print (consec)
		keyboard.press_and_release(str(key_press[int(maximum_indices[0][0])]))
		print(str(key_press[int(maximum_indices[0][0])]))
		consec=0
	numFrames += 1
	if numFrames >= 30:
		if fl ==1:
			print ("Calibration Completed")
			fl=0
	else :
		print (numFrames)
		fl = 1

	# show the frame to our screen
	cv2.imshow("Frame", clone)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()