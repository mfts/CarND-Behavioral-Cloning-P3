import csv
import cv2
import numpy as np

# path to dataset
path1 = "./data/try-7-long/"

def get_data(path):
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # collect the 3 images (left, right, center) for each entry
    images = []
    measurements = []
    for line in lines:
        steering_center = float(line[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        img_center = cv2.imread(path+ 'IMG/' + line[0].split('/')[-1])
        img_left = cv2.imread(path+ 'IMG/' + line[1].split('/')[-1])
        img_right = cv2.imread(path+ 'IMG/' + line[2].split('/')[-1])

        images.extend([img_center, img_left, img_right])
        measurements.extend([steering_center, steering_left, steering_right])

    # mirror the images and the angle data
    augmented_images, augmented_measurements = [], []
    for image,measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*(-1.0))

    return np.array(augmented_images), np.array(augmented_measurements)

X_train, y_train = get_data(path1)

# initialize the keras model

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# use a mean-squared error loss function and an adam optimizer
model.compile(loss='mse', optimizer='adam')

# fit the model with a 20% validation set, over 5 epochs and shuffled data 
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

# save model to file
model.save('model.h5')