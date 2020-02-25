#!/usr/bin/env python
# coding: utf-8

# In[36]:


import os
import random
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing

class_names = ['axial', 'coronal', 'sagittal']

width = 96
height = 96

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.png')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images


images_type_0 = load_images('I:/brain_cnn/axial')
images_type_1 = load_images('I:/brain_cnn/coronal')
images_type_2 = load_images('I:/brain_cnn/sagittal')

plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(images_type_0))
    plt.imshow(image)
    
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

# show the plot
#plt.show()


# In[37]:


X_type_0 = np.array(images_type_0)
X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)

print(X_type_0.shape)
print(X_type_1.shape)
print(X_type_2.shape)


# In[38]:


# one big array containing ALL the images:
X = np.concatenate((X_type_0, X_type_1, X_type_2), axis=0)

X = X / 255.

X.shape

from keras.utils import to_categorical

y_type_0 = [0 for item in enumerate(X_type_0)]
y_type_1 = [1 for item in enumerate(X_type_1)]
y_type_2 = [2 for item in enumerate(X_type_2)]

y = np.concatenate((y_type_0, y_type_1, y_type_2), axis=0)

y = to_categorical(y, num_classes=len(class_names))

print(y.shape)


# In[39]:


from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 30
batch_size = 32
color_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

    model.add(Convolution2D(conv_1, (3, 3),
                            input_shape=(width, height, color_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))
        
    model.add(Flatten())
        
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(len(class_names), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    return model

import numpy as np
np.random.seed(1) # for reproducibility

# model with base parameters
model = build_model()

model.summary()


# In[42]:


epochs = 3

model.fit(X, y, epochs=epochs)

type_2 = preprocessing.image.load_img('H:/FYP/Brain_Tumor_Dataset/no/5.jpeg',
                                      target_size=(width, height))
plt.imshow(type_2)
plt.axis("off")
plt.show()

type_2_X = np.expand_dims(type_2, axis=0)

predictions = model.predict(type_2_X)

print('The type predicted is: {}'.format(class_names[np.argmax(predictions)]))


# In[28]:


model.summary()


# In[29]:


model.save('food_cnn_model.h5')


# In[ ]:




