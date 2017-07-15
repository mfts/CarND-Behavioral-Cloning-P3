import csv
import cv2
import numpy as np

lines = []
with open('./data/try-7-long/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    steering_center = float(line[3])
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    img_center = cv2.imread('./data/try-7-long/IMG/' + line[0].split('/')[-1])
    img_left = cv2.imread('./data/try-7-long/IMG/' + line[1].split('/')[-1])
    img_right = cv2.imread('./data/try-7-long/IMG/' + line[2].split('/')[-1])
    
    images.extend([img_center, img_left, img_right])
    measurements.extend([steering_center, steering_left, steering_right])
    
augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers import MaxPooling2D

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

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')