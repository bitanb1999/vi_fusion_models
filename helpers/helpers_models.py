#!/usr/bin/env python
# coding: utf-8

# Training CNNs to predict turnout for Kenya 2013
# Helpers to save the models over which we are iterating
# Chris Arnold
# November 2020


# from keras.utils import to_categorical
# from keras import models
# from keras import layers
# from keras import optimizers
# from keras import regularizers
# from keras.layers.normalization import BatchNormalization
#
#
#
# #-- Very Easy Toy Model -------------------------------------------------------------
# def model_0(train_images, input_shape):
#     ''' Very easy model to check code'''
#     network = models.Sequential()
#     # L1
#     network.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',
#                               input_shape=input_shape))
#     # network.add(layers.MaxPooling2D((2, 2)))
#     # L4
#     network.add(layers.Flatten())
#     network.add(layers.Dense(16, activation='relu', input_shape=train_images.shape[1:4]))
#     network.add(layers.Dense(1, activation = 'sigmoid'))
#     return(network)


from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU




def model1():
    # new model
    network = Sequential()
    #--- Convolutional Layers
    # l1
    network.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(96, 96, 7)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # l2
    network.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # l3
    network.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # l4
    network.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # l5
    network.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # l6
    network.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    # network.add(MaxPooling2D((2, 2)))
    #--- Dense Layers
    network.add(Flatten())
    # Dense 1
    network.add(Dense(128, activation='relu'))
    network.add(Dropout(.4))
    # Dense 2
    network.add(Dense(32, activation='relu'))
    network.add(Dropout(.3))
    # Dense 3
    network.add(Dense(1, activation='sigmoid'))
    return(network)



def model2():
    '''Insight: Going deeper than 5 layers here stops the learning '''
    # new model
    network = Sequential()
    #--- Convolutional Layers
    # Convolutional Layer
    network.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 7)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    #--- Dense Layers
    # Dense Layer
    network.add(Dropout(.3))
    network.add(Flatten())
    network.add(Dense(512, activation='relu'))
    # Dense Layer
    network.add(Dropout(.4))
    network.add(Dense(128, activation='relu'))
    # Dense 3
    network.add(Dense(1, activation='sigmoid'))
    return(network)



def model3():
    # new model
    network = Sequential()
    #--- Convolutional Layers
    # Convolutional Layer
    network.add(Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 7)))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    # Convolutional Layer
    network.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D((2, 2)))
    #--- Dense Layers
    # Dense Layer
    network.add(Dropout(.3))
    network.add(Flatten())
    network.add(Dense(512, activation='relu'))
    # Dense Layer
    network.add(Dropout(.4))
    network.add(Dense(128, activation='relu'))
    # Dense Layer
    network.add(Dense(1, activation='sigmoid'))
    return(network)
