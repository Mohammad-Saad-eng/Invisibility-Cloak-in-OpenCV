#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


cap = cv2.VideoCapture(0)
cv2.namedWindow('HP', cv2.WINDOW_NORMAL)
backg = 0

for i in range(30):
    ret, backg = cap.read()
    
backg = cv2.flip(backg,1)

while 1:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    blur = cv2.GaussianBlur(hsv, (35,35),0)
    
    lower_red = np.array([59, 122, 0])
    upper_red = np.array([110, 255, 167])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations = 3)
    
    frame[np.where(mask==255)] = backg[np.where(mask==255)]
    
    cv2.imshow('HP',frame)
    
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

