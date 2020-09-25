#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[28]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Open Image

# In[29]:


img = cv2.imread('Images/eye_face.jpg')


# ### Show Image

# In[30]:


plt.imshow(img)


# ### Fix Image

# In[31]:


fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ### Show Image

# In[32]:


plt.imshow(fix_img)


# ### Create Classifier

# In[33]:


eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')


# ### It's a Kind of Magic

# In[34]:


def detect_eyes(fix_img):
    eyes_rects = eye_classifier.detectMultiScale(fix_img)
    
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(fix_img,
                     (x,y),
                     (x+w, y+h),
                     (255,255,255),
                     10)
    return fix_img    


# ### Show Results

# In[35]:


result = detect_eyes(fix_img)
plt.imshow(result)


# In[ ]:




