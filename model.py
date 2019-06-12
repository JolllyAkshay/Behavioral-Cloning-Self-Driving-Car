import os
import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                
                for i in range(3):
                    current_path = 'data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if i == 0:
                        measurement = measurement
                        
                    elif i == 1:
                        measurement = measurement + 0.2
                    else:
                        measurement = measurement - 0.2
            
                    measurements.append(measurement)
					
				
            #for image, measurement in zip(images, measurements):  
             #   images.append(cv2.blur(image, (5,5)))
              #  measurements.append(measurement)
        
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)    

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
			
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D

# Define model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
#model.compile(optimizer='adam', loss='mse')

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,steps_per_epoch=len(train_samples)/batch_size,validation_data=validation_generator,validation_steps=(len(validation_samples)/batch_size),epochs=10, verbose=1)

model.save('model.h5')