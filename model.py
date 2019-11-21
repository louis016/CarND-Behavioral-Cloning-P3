# import useful libraries
import os
import csv
import cv2
import sklearn
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from random import shuffle

# import image preprocessing function
from utils import INPUT_SHAPE, batch_generator

# import keras library 
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D
from keras.layers import Cropping2D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

# model building
def nvidia16(args):
    '''
    Nvidia end-to-end model
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    '''
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):

    # Save the best model after every epoch
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    # Use mean square error to estimate the difference between expected steering
    # angle and actual steering angle
    model.compile(loss='mse', optimizer = Adam(lr=args.learning_rate))

    # Trains the model on data generated batch-by-batch by a Python generator
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),   
                        samples_per_epoch = args.samples_per_epoch,
                        verbose=1,
                        callbacks = [checkpoint],
                        max_q_size=1,
                        validation_data = batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples = len(X_valid),
                        nb_epoch=args.nb_epoch)

def main():
    
    # Parser for command-line options, arguments and sub-commands
    parse = argparse.ArgumentParser(description='Car Behavioral Cloning Project')
    parse.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parse.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parse.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=5)
    parse.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parse.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parse.add_argument('-b', help='batch sizes', dest='batch_size', type=int, default=40)
    parse.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parse.parse_args()    
    print(args)

    #load data
    data = load_data(args)
    #build model
    model = nvidia16(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, *data)


if __name__ == "__main__":
    main()