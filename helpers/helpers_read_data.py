#!/usr/bin/env python
# coding: utf-8

# Training CNNs to predict turnout for Kenya 2013
# Helpers to read in the data
# Chris Arnold
# November 2020

import glob
import rasterio
import numpy as np
# import pyreadr
# import pandas as pd
from itertools import compress
import math
import random
import json
import math
import time



# Reading Sat Images
def read_stllt_images_and_names(images_path):
    ''' Reads in satellite images from folder. Requires rasterio to
    take care of images with more than just RGB channels'''
    n = glob.glob(images_path+'/'+'*.tif')
    stllt_images = []
    stllt_images_names = []
    for fname in n:
        # creates polling station id from file name
        stllt_images_names.append(fname.replace((images_path+'/'), '').replace('.tif', ''))
        # gets image
        image_constr = rasterio.open(fname)
        # Reading as array
        # 96 is great for learning, because you can devide it easily
        image = image_constr.read()[:,2:98,2:98]
        stllt_images.append(image)
    # Correct the location of the bands to last
    stllt_images= np.moveaxis(stllt_images, 1,3)
    # Normalise the data
    minval = np.min(stllt_images)
    maxval = np.max(stllt_images)
    stllt_images_normalized = (stllt_images - minval)/(maxval - minval)
    return(stllt_images_normalized, stllt_images_names)




# Joins
def join_sat_imgs_ps_data_at_stream_level(image_ps_id, image_data, ps_id, str_id, turnout):
    '''joins sat images and polling station level data at stream level.
    '''
    # 1 Prepare data dictionaries for easy querying
    # Images dict at PS level
    images_data_dict = dict(zip(image_ps_id,image_data))
    # PS data dict at stream level
    dat_votes = []
    for i in range(len(ps_id)):
        one_line = [ps_id[i], str_id[i], turnout[i]]
        dat_votes.append(one_line)
    str_data_dict = dict(zip(str_id, dat_votes))
    # 2 Select Data
    # 2.1 Select PS data on the basis of unique IDs for streams. nota bene: issue is handled in str_data_dict already
    # We are doing this because there are 29453 - 29445 = 8 data entry mistakes:
    # print(len(str_id))
    # print(len(set(str_id)))
    unique_str_id = list(set(str_id))
    ps_id_at_str_level = []
    turnout_at_str_level = []
    str_id_at_str_level = []
    for i in range(len(unique_str_id)):
        # polling_station
        ps_id_at_str_level.append(str_data_dict[unique_str_id[i]][0])
        # Polling Stream
        str_id_at_str_level.append(str_data_dict[unique_str_id[i]][1])
        # turnout
        turnout_at_str_level.append(str_data_dict[unique_str_id[i]][2])
    # 2.2 Select image data on the basis of unique IDs for streams and correct for the missing satellite images
    images_at_str_level_wona = []
    mask = [True]*len(unique_str_id)
    for i in range(len(unique_str_id)):
        try:
            images_at_str_level_wona.append(images_data_dict[ps_id_at_str_level[i]])
        except:
            mask[i] = False

    missing_images = len(mask) - np.sum(mask)
    # 2.3 select ps data on the basis of missing sat images
    ps_id_at_str_level_wona = list(compress(ps_id_at_str_level, mask))
    turnout_at_str_level_wona = list(compress(turnout_at_str_level, mask))
    str_id_at_str_level_wona = list(compress(str_id_at_str_level, mask))

    return(images_at_str_level_wona, ps_id_at_str_level_wona, turnout_at_str_level_wona, str_id_at_str_level_wona)





def make_train_test_for_observed_streams(
    images_at_str_level_wona, turnout_at_str_level_wona,
    str_id_at_str_level_wona, observed_streams, which_experiment):
    '''
    Joins the polling station data and the satellite images.
    Selects only those data for which we have observed values'''
    # which streams were observed?
    obs_str_at_str_level_wona = [stream in set(observed_streams) for stream in str_id_at_str_level_wona]
    # make an array mask from it to Select observed stations
    mask_observed = np.array([i==1 for i in obs_str_at_str_level_wona])

    # generate a stack of images at stream level
    image_data_all_bands = np.stack(images_at_str_level_wona, axis = 0)

    # Define training and test and val set sizes
    observed_ps_set_size = np.sum(obs_str_at_str_level_wona)
    testval_set_size = math.floor(.1*observed_ps_set_size)
    train_set_size = observed_ps_set_size - 2*testval_set_size
    # Select observed images
    observed_ps_images = image_data_all_bands[mask_observed,:,:,:].reshape((observed_ps_set_size, 96, 96, 7))
    # Select observed labels
    observed_ps_labels = list(compress(turnout_at_str_level_wona, mask_observed))
    observed_str_id_values = list(compress(str_id_at_str_level_wona, mask_observed))
    # randomize the order
    new_order = random.sample(range(observed_ps_set_size), observed_ps_set_size)
    # randomize sat images
    train_images = observed_ps_images[new_order[0:train_set_size],:,:,:].reshape(train_set_size, 96, 96, 7)
    val_images = observed_ps_images[new_order[train_set_size:(train_set_size+testval_set_size)],:,:,:].reshape((testval_set_size, 96, 96, 7))
    test_images = observed_ps_images[new_order[(train_set_size+testval_set_size):observed_ps_set_size],:,:,:].reshape((testval_set_size, 96, 96, 7))
    # randomize labels
    train_labels = [observed_ps_labels[i] for i in new_order[0:train_set_size]]
    val_labels = [observed_ps_labels[i] for i in new_order[train_set_size:(train_set_size+testval_set_size)]]
    test_labels = [observed_ps_labels[i] for i in new_order[(train_set_size+testval_set_size):observed_ps_set_size]]
    # write the train_val_test split as json for eventual later use
    train_str_id = [observed_str_id_values[i] for i in new_order[0:train_set_size]]
    val_str_id = [observed_str_id_values[i] for i in new_order[train_set_size:(train_set_size+testval_set_size)]]
    test_str_id = [observed_str_id_values[i] for i in new_order[(train_set_size+testval_set_size):observed_ps_set_size]]
    with open('best_models/'+which_experiment+'/train_val_test_stream_ids.json', 'w', encoding='utf-8') as f:
        json.dump([train_str_id, val_str_id, test_str_id], f, ensure_ascii=False, indent=4)
    return(train_images, val_images, test_images, train_labels, val_labels, test_labels)


def make_train_test_for_unobserved_streams(
    images_at_str_level_wona, turnout_at_str_level_wona,
    str_id_at_str_level_wona, observed_streams, which_experiment):
    '''
    Joins the polling station data and the satellite images.
    Selects only those data for which we have UNobserved values'''
    # which streams were observed?
    obs_str_at_str_level_wona = [stream in set(observed_streams) for stream in str_id_at_str_level_wona]
    # make an array mask from it to Select observed stations
    mask_observed = np.array([i==1 for i in obs_str_at_str_level_wona])
    mask_unobserved = np.array([i==0 for i in obs_str_at_str_level_wona])
    # generate a stack of images at stream level
    print('The length of images_at_str_level_wona', len(images_at_str_level_wona))
    image_data_all_bands = np.stack(images_at_str_level_wona, axis = 0)
    # calc some helpful sizes
    observed_ps_set_size = np.sum(obs_str_at_str_level_wona)
    unobserved_ps_set_size = len(obs_str_at_str_level_wona) - observed_ps_set_size
    testval_set_size = math.floor(.1*unobserved_ps_set_size)
    train_set_size = unobserved_ps_set_size - 2*testval_set_size

    # Unobserved selection
    # Select unobserved images
    unobserved_ps_images = image_data_all_bands[mask_unobserved,:,:,:].reshape((unobserved_ps_set_size, 96, 96, 7))
    # Select unobserved labels
    unobserved_ps_labels = list(compress(turnout_at_str_level_wona, mask_unobserved))
    unobserved_str_id_values = list(compress(str_id_at_str_level_wona, mask_unobserved))


    # randomize the order
    new_order = random.sample(range(unobserved_ps_set_size), unobserved_ps_set_size)
    # randomize sat images
    train_images = unobserved_ps_images[new_order[0:train_set_size],:,:,:].reshape(train_set_size, 96, 96, 7)
    val_images = unobserved_ps_images[new_order[train_set_size:(train_set_size+testval_set_size)],:,:,:].reshape((testval_set_size, 96, 96, 7))
    test_images = unobserved_ps_images[new_order[(train_set_size+testval_set_size):unobserved_ps_set_size],:,:,:].reshape((testval_set_size, 96, 96, 7))
    # randomize labels
    train_labels = [unobserved_ps_labels[i] for i in new_order[0:train_set_size]]
    val_labels = [unobserved_ps_labels[i] for i in new_order[train_set_size:(train_set_size+testval_set_size)]]
    test_labels = [unobserved_ps_labels[i] for i in new_order[(train_set_size+testval_set_size):unobserved_ps_set_size]]
    # write the train_val_test split as json for eventual later use
    train_str_id = [unobserved_str_id_values[i] for i in new_order[0:train_set_size]]
    val_str_id = [unobserved_str_id_values[i] for i in new_order[train_set_size:(train_set_size+testval_set_size)]]
    test_str_id = [unobserved_str_id_values[i] for i in new_order[(train_set_size+testval_set_size):unobserved_ps_set_size]]
    with open('best_models/'+which_experiment+'/train_val_test_stream_ids.json', 'w', encoding='utf-8') as f:
        json.dump([train_str_id, val_str_id, test_str_id], f, ensure_ascii=False, indent=4)
    return(train_images, val_images, test_images, train_labels, val_labels, test_labels)


# Reading and preparing data
def read_and_prep_data(images_path, ps_data_path, which_experiment, observed = True):
    '''Binder for the functions above.
    Prepares all data for the observed stations usually.
    '''
    t1 = time.time()
    print('Starting with the data preparation')
    ## read data
    print('Reading satellite images')
    # read all sat images into one list
    image_data, image_ps_id = read_stllt_images_and_names(images_path)
    # read other vars
    print('Reading all other ps vars')
    with open(ps_data_path) as json_file:
        variables_for_saving = json.load(json_file)

    ps_id = variables_for_saving[0]
    str_id = variables_for_saving[1]
    turnout = variables_for_saving[2]
    observed_streams = variables_for_saving[3]
    ## Bring the sat data and the polling station data together
    images_at_str_level_wona, ps_id_at_str_level_wona, turnout_at_str_level_wona, str_id_at_str_level_wona \
        = join_sat_imgs_ps_data_at_stream_level(image_ps_id, image_data, ps_id, str_id, turnout)
    ## Prepping Data for Experiment
    # Identify the Satellite Images that have a observed Polling Station
    print('Now working on train-val-test split')
    if observed == True:
        train_images, val_images, test_images, train_labels, val_labels, test_labels = make_train_test_for_observed_streams(images_at_str_level_wona, turnout_at_str_level_wona,
            str_id_at_str_level_wona, observed_streams, which_experiment)
    if observed == False:
        train_images, val_images, test_images, train_labels, val_labels, test_labels = make_train_test_for_unobserved_streams(images_at_str_level_wona, turnout_at_str_level_wona,
            str_id_at_str_level_wona, observed_streams, which_experiment)
    t2 = time.time()
    print('Done with data prep. It took me', round(t2-t1, 0), 'seconds')
    return(train_images, val_images, test_images, train_labels, val_labels, test_labels)
