#/**************************************************************************************************************

# BTP PROJECT
# TEAM MEMBERS- ANMOL CHHABRA (B15CS009)
			  # HARSHIT SINGH (B15CS019)

#/**************************************************************************************************************
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

def nothing(x):
	pass
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=16,
	help="max buffer size")
args = vars(ap.parse_args())

# /**************************************************************************************************************/

whiteLower = (20,15,175)
whiteUpper = (75,85,255)




#/***************************************************************************************************************




pts = deque(maxlen=args["buffer"])

if not args.get("video", False):
	camera = cv2.VideoCapture(0)


else:
	camera = cv2.VideoCapture(args["video"])

while True:
	# grab the current frame

	# /******************************/
	# /******************************/

	(grabbed, frame) = camera.read()


	if args.get("video") and not grabbed:
		break

	
	frame = imutils.resize(frame, width=600)
	cv2.imshow("result",frame)
	# blurred = cv2.medianBlur(frame,13)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	
	mask = cv2.inRange(hsv, whiteLower, whiteUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		 
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		
		if (radius<15):
			
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 3)
			cv2.circle(frame, center, 1, (0, 0, 255), -1)

	
	for i in xrange(1, len(pts)):
		
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)

		# if (radius<15):
		# 	cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(0) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
