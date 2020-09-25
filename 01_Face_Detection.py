#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Create Classifier

# In[3]:


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')


# ### Open Image

# In[5]:


image = cv2.imread('Images/eye_face.jpg')
fix_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)


# ### Detect Faces

# In[7]:


faces = face_classifier.detectMultiScale(image, 1.3, 5)


# ### No Faces Detected

# In[8]:


if faces is ():
    print('No Faces found')


# ### It's a kind of Magic

# In[11]:


def detect_face(fix_img):
    face_rects = face_classifier.detectMultiScale(fix_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(fix_img,
                     (x,y),
                     (x+w, y+h),
                     (255,0,0),
                     10)
        
    return fix_img    


# In[12]:


result = detect_face(fix_img)
plt.imshow(result)


# In[ ]:




