#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Create Classifiers

# In[38]:


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')


# ### Open Image

# In[39]:


img = cv2.imread('Images/eye_face.jpg')


# ### Fix Image

# In[40]:


fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ### Face Classifier

# In[41]:


faces = face_classifier.detectMultiScale(fix_img, 1.3, 5)


# ### If No Face Detected

# In[42]:


if faces is ():
    print('No Faces found')


# ### It's a Kind of Magic

# In[43]:


def detect_faces_eyes(fix_img):
    
    face_rects = face_classifier.detectMultiScale(fix_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(fix_img,
                     (x,y),
                     (x+w, y+h),
                      (255,0,0),
                      7)
        eyes_rects = eye_classifier.detectMultiScale(fix_img)
        for (ix,iy,iw,ih) in eyes_rects:
            cv2.rectangle(fix_img,
                         (ix,iy),
                         (ix+iw, iy+ih),
                         (0,0,255),
                         5)
    return fix_img        


# In[ ]:


result = detect_faces_eyes(fix_img)
plt.imshow(result)


# In[ ]:





# In[ ]:




