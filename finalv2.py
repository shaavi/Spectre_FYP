#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage import measure

#read image
image = cv2.imread('H:/FYP/interim/images/1.jpg',0)

#preprocess using median blur
img = cv2.medianBlur(image,5)

plt.imshow(img)


# In[30]:


#erosion using morphological operations

ret,th1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(th1, kernel, iterations = 4)
plt.imshow(erosion)


# In[31]:


# perform a connected component analysis on the eroded image, then initialize a mask to store only the "large" components

labels = measure.label(erosion, neighbors=8, background=0)
mask1 = np.zeros(erosion.shape, dtype="uint8")
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(erosion.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 30000:
		mask1 = cv2.add(mask1, labelMask)
        
plt.imshow(mask1)


# In[32]:


mask = cv2.dilate(mask1,kernel,iterations = 6)
plt.imshow(mask)


# In[33]:


#extract the brain using the mask
img = io.imread("H:/FYP/interim/images/1.jpg")
mask2 = np.where((mask<200),0,1).astype('uint8')

brain_img = img*mask2[:,:,np.newaxis]

plt.imshow(brain_img)


# In[35]:


# Load image, grayscale, Otsu's threshold, and extract ROI
image = brain_img
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
x,y,w,h = cv2.boundingRect(thresh)
ROI = image[y:y+h, x:x+w]
plt.imshow(ROI)


# In[36]:


# Color segmentation on ROI
hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 152])
upper = np.array([179, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
plt.imshow(mask)


# In[37]:


# Crop left and right half of mask
x, y, w, h = 0, 0, ROI.shape[1]//2, ROI.shape[0]
left = mask[y:y+h, x:x+w]
right = mask[y:y+h, x+w:x+w+w]

# Count pixels
left_pixels = cv2.countNonZero(left)
right_pixels = cv2.countNonZero(right)

print('Left pixels:', left_pixels)
print('Right pixels:', right_pixels)


# In[38]:


# Crop top and bottom half of mask
x, y, w, h = 0, 0, mask.shape[0], mask.shape[1]//2
bottom = mask[y+h:y+h+h, x:x+w]
top = mask[y:y+h, x:x+w]

# Count pixels
top_pixels = cv2.countNonZero(top)
bottom_pixels = cv2.countNonZero(bottom)

print('Top pixels:', top_pixels)
print('Bottom pixels:', bottom_pixels)


# In[39]:


if left_pixels > right_pixels:
    print("Left")
else:
    print("Right")
    
if top_pixels > bottom_pixels:
    print("Top")
else:
    print("Bottom")


# In[ ]:




