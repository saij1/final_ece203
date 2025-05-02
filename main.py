#https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

import cv2
import handtrackingmodulesai as htm
import serial

#initialize serial connection
ser1 = serial.Serial('/dev/cu.usbmodem14101',9600) #change according to your arduino COM port
ser1.timeout = 1

#initialize video feed with laptop camera
cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

#specify video codec and save video feed to file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

#initialize hand detector object with 1 hand tracking and minimum detection confidence of 0.75
detector = htm.handDetector(detectionCon = 0.75, maxHands = 1)

while True:
    ret, img = cam.read()
    #processes frame, runs ml model and draws landmark
    img = detector.findHands(img)

    # finds positions of landmarks
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # thumb tip (lm 4) to the left of 
        # thumb knuckle (lm 3) indicates thumb is outstretched
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
        # ids of landmarks of tips of index through pinky
        for id in [8, 12, 16, 20]:
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(100+totalFingers)
        ser1.write(str(100+totalFingers).encode())
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 250), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    out.write(img)


    cv2.imshow('Camera', img)



    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
out.release()
cv2.destroyAllWindows()
ser1.close()