import sys
import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2  
import matplotlib.image as mpimg
import numpy as np
import os

path="I:/BBBB/final_output_data/"
direct = "I:/brain_cnn/axial_skull strip/"
brain = []
for img1 in glob.glob(path+'*.jpg'):
    img = cv2.imdecode(np.fromfile(img1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #img = np.array(img).astype()
    # apply median filter
    img_out = cv2.medianBlur(img,5)
    
    #otsu thresholding
    #_, binarized = cv2.threshold (img_out, 55, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    
    #erode
    ret,th1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    
    erosion = cv2.erode(th1, kernel, iterations = 5)
    plt.imshow(erosion)
    
    # largest connected component
    foreground_value = 255
    mask = np.uint8(erosion == foreground_value)
    labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
    largest_label = 1+np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    mask1 = np.zeros_like(erosion)
    mask1[labels == largest_label] = foreground_value
    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize = (4, 4))
    plt.imshow(mask1)
    
    #dilate
    mask = cv2.dilate(mask1,kernel,iterations = 6)
    plt.imshow(mask)
    
    
    #masking
    brain_img = np.zeros_like(img)
    brain_img[mask == 255] = img[mask == 255]
    #plt.imshow(brain_img)
    
    
    
    #img=mpimg.imread(img)
    #mgplot = plt.imshow(img)
    head, tail = os.path.split(img1)

    print(os.path.basename(img1))

    brain.append([brain_img, tail])
    #cv.imwrite(output_folder + output_name, output)
    cv2.imwrite(direct + os.path.basename(img1), brain_img)
    plt.imshow(brain_img)
    plt.show()
    
    
#for i in range(20):
    #image = preprocessing.image.array_to_img(random.choice(brain))

 #   plt.subplot(4,5,i+1),plt.imshow(brain[i][0],'gray')
  #  plt.axis('off')
   # plt.title(brain[i][1])

    
   
