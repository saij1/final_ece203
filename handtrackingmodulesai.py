#https://lvimuth.medium.com/hand-detection-in-python-using-opencv-and-mediapipe-30c7b54f5ff4
#https://circuitdigest.com/microcontroller-projects/gesture-based-intelligent-appliance-control

import cv2 #cv2 for laptop camera and passing video feed to mediapipe
import mediapipe as mp #mediapipe ML models for palm detection and hand landmark recognition
import time

class handDetector():
    #mode = False is for treating images like video stream
    #detectionCon for palm detection minimum confidence
    #trackCon for hand landmark minimum confidence
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # un mirror the image
        img = cv2.flip(img, 1)
        # specify color space
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process frame with ML pipeline and store results
        self.results = self.hands.process(imgRGB)

        #checks if there are any landmarks and then draws the landmarks if any are present
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        #check if any landmarks are tracked
        if self.results.multi_hand_landmarks:
            #store landmarks for first hand (we are only using right hand info)
            myHand = self.results.multi_hand_landmarks[handNo]
            #iterate through landmarks for this hand
            for id, lm in enumerate(myHand.landmark):
                #store dimensions
                height, width, channels, = img.shape
                #scale up landmark detection to image dimensions
                posx, posy = int(lm.x * width), int(lm.y * height)
                #append landmarks to list of landmarks
                lmList.append([id, posx, posy])
                if draw:
                    cv2.circle(img, (posx, posy), 5, (255,0,255), cv2.FILLED)
        #return list of landmarks
        return lmList