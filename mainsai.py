#https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

import cv2
import time
import handtrackingmodulesai as htm
import serial

#ser1 = serial.Serial('COM8', 9600)
#ser1.timeout = 1
cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
detector = htm.handDetector(detectionCon = 0.75, maxHands = 1)

while True:
    ret, img = cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # thumb tip (lm 4) to the left of 
        # thumb knuckle (lm 3) indicates thumb is outstretched
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
        for id in [8, 12, 16, 20]:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        fingerCount = fingers.count(1)
        cv2.rectangle(img, (20,255), (170,425), (0,255,250), cv2.FILLED)
        cv2.putText(img, str(fingerCount), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
    out.write(img)


    cv2.imshow('Camera', img)



    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()