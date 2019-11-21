# use csv to load the csv data file
import csv
import numpy as np
# use open-cv to load and process the images
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers import Conv2D, MaxPooling2D

input_dir = './data/driving_log.csv'
input_shape = (160, 320, 3)

# Help Funtions
def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Load data and split into training data and validation data
def load_data(input_dir):

    data_list = []

    with open(input_dir, 'r') as f:
        data = csv.reader(f)
        for num, line in enumerate(data):
            if(num!=0):
                data_list.append(line)

    images = []
    steering_angles = []

    for sample in data_list:
        image_path = './data/IMG/' + sample[0].split('/')[-1]
        angle = float(sample[3])
        center = cv2.imread(image_path)

        images.append(center)
        steering_angles.append(angle)

    '''
    augmented_images = []
    augmented_angles = []
    # Data Augmentation
    for image, steering_angle in zip(images, steering_angles):
        augmented_images.append(image)
        augmented_angles.append(steering_angle)
        image, steering_angle = random_flip(image, steering_angle)
        image, steering_angle = random_translate(image, steering_angle, range_x=100, range_y=10)
        image = random_brightness(image)
        augmented_images.append(image)
        augmented_angles.append(steering_angles)
    '''    
    X_train = np.asarray(images)
    y_train = np.asarray(steering_angles)

    return X_train, y_train


# First model
def model(input_shape):

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))
    model.summary()
    
    return model

# Lenet model
def lenet(input_shape):

    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (60,60))))
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.summary()

    return model 

# Nvidia model
def nvidia16(input_shape):

    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


# load data and train model
X_train, y_train = load_data(input_dir)
model = model(input_shape)
model.compile(loss= 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2,verbose = 1, shuffle = True, nb_epoch = 5)
model.save('./model/model3.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()