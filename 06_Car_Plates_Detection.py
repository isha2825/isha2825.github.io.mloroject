#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[23]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Open Image

# In[24]:


img = cv2.imread('Images/car_plate.jpg')


# ### Convert Image

# In[25]:


def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)


# ### Display it

# In[26]:


display(img)


# ### Create Classifier

# In[27]:


plate_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_russian_plate_number.xml')


# ### It's a Kind of Magic

# In[31]:


def detect_plate(img):
    
    plate_img = img.copy()
    
    plate_rects = plate_classifier.detectMultiScale(plate_img, 1.1, 1)
    
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,
                     (x,y),
                     (x+w, y+h),
                     (255,255,255),
                     5)
    return plate_img    


# ### Check the Results

# In[32]:


result = detect_plate(img)


# In[30]:


display(result)


# In[ ]:




