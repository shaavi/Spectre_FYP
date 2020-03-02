#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import random
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing

get_ipython().run_line_magic('matplotlib', 'inline')

class_names = ['yes', 'no']

width = 256
height = 256

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.jpg')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path,
                                             target_size=(width, height))
        x = preprocessing.image.img_to_array(image)

        images.append(x)
    return images

images_type_0 = load_images('I:/BBBB/yes_no/train/yes')
images_type_1 = load_images('I:/BBBB/yes_no/train/no')

X_type_0 = np.array(images_type_0)
X_type_1 = np.array(images_type_1)

# one big array containing ALL the images:
X = np.concatenate((X_type_0, X_type_1), axis=0)

X = X / 255.

X.shape


# In[12]:


from keras.utils import to_categorical

y_type_0 = [0 for item in enumerate(X_type_0)]
y_type_1 = [1 for item in enumerate(X_type_1)]

y = np.concatenate((y_type_0, y_type_1), axis=0)

y = to_categorical(y, num_classes=len(class_names))

print(y.shape)


# In[13]:


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


# In[14]:


import numpy as np
np.random.seed(1) # for reproducibility

# model with base parameters
model = build_model()

model.summary()


# In[15]:


epochs = 10


# In[16]:


model.fit(X, y, epochs=epochs)


# In[17]:


model.summary()


# In[18]:


model.save('yes_no_final.h5')


# In[ ]:




