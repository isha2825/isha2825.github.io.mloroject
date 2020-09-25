#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[3]:


import cv2
import numpy as np


# ### Create Classifier

# In[4]:


car_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')


# ### Capture Video

# In[5]:


cap = cv2.VideoCapture('Video/Vehicles.mp4')


# ### It's a Kind of Magic

# In[ ]:


# while Loop
while cap.isOpened():


    # Read the capture
    ret, frame = cap.read()
    
    # Pass the Frame to the Classifier
    cars = car_classifier.detectMultiScale(frame, 1.4, 2)
    
    # for Loop
    for (x,y,w,h) in cars:
    
        # Bound Boxes to Identified Bodies
        
        cv2.rectangle(frame,
                         (x,y),
                         (x+w, y+h),
                         (0,225,225),
                         2)
        cv2.imshow('Cars', frame)
        
        
        
    # Exit with Esc button
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    
# Release the Capture & Destroy All Windows
cap.release()
cv2.destroyAllWindows()



# In[ ]:




