#!/usr/bin/env python
# coding: utf-8

# Training CNNs to predict turnout for Kenya 2013
# Helpers to check the performance of the model
# Chris Arnold
# November 2020


import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
import numpy as np
import csv

import h5py

from helpers.helpers_models import *
# import matplotlib.Figure.set_size_inches

def performance_plotter(history, filename, best_val_mae):
    # 1 Prepare data
    history_dict = history.history
    # read hist data
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    mae = history_dict['mae']
    val_mae = history_dict['val_mae']
    epochs = range(1, len(loss) + 1)
    # Define burnin
    # burnin = 1
    # if burnin_val != 0:
    #     burnin = np.floor(len(loss)*burnin_val)
    # print(burnin)
    # 2 Plot
    f, axarr = plt.subplots(2)
    # Subplot 1
    axarr[0].plot(epochs, loss, 'b--', label='Training loss')
    axarr[0].plot(epochs, val_loss, 'g', label='Validation loss')
    # axarr[0].set_xlim([burnin-1, (len(loss)+1)])
    axarr[0].set_ylabel('Loss')
    axarr[0].legend(fancybox=True, framealpha=0)
    # Subplot 2
    axarr[1].plot(epochs, mae, 'b--', label='Training MAE')
    axarr[1].plot(epochs, val_mae, 'g', label='Validation MAE')
    axarr[1].text(2, (best_val_mae+.03), ('Best MAE on Val Set = '+str(round(best_val_mae,5))))
    # axarr[1].set_xlim([burnin-1, (len(loss)+1)])
    axarr[1].set_xlabel('Epochs')
    axarr[1].set_ylabel('MAE')
    axarr[1].legend(fancybox=True, framealpha=0)

    plt.savefig(filename, bbox_inches='tight', transparent=True)



def train_models(model_name, batch_size, optimizer, learning_rate,
    train_images, train_labels, val_images, val_labels, which_experiment, epochs = 10):
    ''' trains a model and saves its best version.
    Automatically stops after there is no progress for (here) 30 iterations
    '''
    # Select model
    if model_name == 'model1':
        network = model1()
    elif model_name == 'model2':
        network = model2()
    elif model_name == 'model3':
        network = model3()
    elif model_name == 'model_inception_V3':
        network = model_inception_V3()
    else: print('Did you mistype the model you wanted to chose?')
    # writing output
    specs_name = (model_name+'_'+str(batch_size)+'_'+optimizer+'_'+str(learning_rate))
    best_model_path = ("best_models/"+which_experiment+'/'+specs_name+"_weights.h5")
    plot_path = ("best_models/"+which_experiment+'/'+specs_name+"_figure.png")
    # set optimizer
    if optimizer == 'adam':
        if learning_rate == 'default':
            optimizer = Adam()
        else:
            optimizer = Adam(lr=learning_rate)
    if optimizer == 'rmsprop':
        if learning_rate == 'default':
            optimizer = RMSprop()
        else :
            optimizer = RMSprop(lr=learning_rate)
    #  Compile
    network.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['mae'])
    # select best models, but with early stopping
    early_stopper = EarlyStopping(monitor='val_mae', patience=75,
        mode='min', min_delta=0.0001)
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_mae', verbose=1,
        save_best_only=True, mode='min')
    callbacks_list = [early_stopper, checkpoint]

    # This is for jupyter notebook
    # https://www.kaggle.com/product-feedback/41221
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # do the work
    history = network.fit(train_images, train_labels,
                epochs=epochs , batch_size=batch_size, callbacks=callbacks_list,
                validation_data=(val_images, val_labels))

    # Valiation performance on the best model
    # Select model
    if model_name == 'model1':
        best_network = model1()
    elif model_name == 'model2':
        best_network = model2()
    elif model_name == 'model3':
        best_network = model3()
    elif model_name == 'model_inception_V3':
        best_network = model_inception_V3()
    else: print('Did you add the new model also to the call for the best model?')
    best_network.load_weights(best_model_path)
    best_network.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['mae'])
    val_mse, val_mae = best_network.evaluate(val_images, val_labels)
    ## Make a visualizer for training performance
    performance_plotter(history, plot_path, val_mae)
    # save results
    val_performance_path = ('best_models/'+which_experiment+'/'+str(val_mae)+'_'+specs_name+'_best_val_performance.csv')
    with open(val_performance_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(zip([val_mse], [val_mae]))
    return(val_mse, val_mae)
