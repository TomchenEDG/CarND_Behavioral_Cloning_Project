
# coding: utf-8

# In[1]:


import csv
import os
import cv2
import sklearn
import numpy as np


# ## 1.Access data path

# In[2]:


samples = []

with open(r'./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# In[3]:


# Divide the data into 80% training data and 20% validate the data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[4]:


len(train_samples),len(validation_samples)


# ## 2.Obtain training and label data

# In[5]:


# The development generator imports the data
def generator(samples, batch_size=32, train=True):
    
    num_samples = len(samples)
    images_and_angles = [
        (0, 0.0),    # central camera
        (1, 0.2),   # left camera
        (2, -0.2),  # right camera
        ]
    
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for batch_sample_index, angle_adjust in images_and_angles:
                    name = './data/IMG/' + batch_sample[batch_sample_index].split('\\')[-1]
                    center_lr_image = cv2.imread(name)
                    center_lr_angle = float(batch_sample[3]) + angle_adjust
                    images.append(center_lr_image)  # center、left、right camera image
                    angles.append(center_lr_angle)  # center、left、right camera angles

            # Data Augmentation
            if train:
                augmented_images, augmented_angles = [], []
                for image, angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    augmented_images.append(cv2.flip(image, 1)) # flip the images horizontally
                    augmented_angles.append(angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[6]:


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[ ]:


next(validation_generator)


# ## 3.Build a model

# In[7]:


from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Activation, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D


# In[10]:


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(
        generator=train_generator,
        samples_per_epoch=len(train_samples)*3*2,   # 3 images per sample * 3 and augmentation data * 2
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples)*3*2,
        nb_epoch=3)

model.save('model.h5')


# In[12]:


model.summary()


# In[ ]:




