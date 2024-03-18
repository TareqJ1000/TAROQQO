# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:18:50 2024

@author: tjaou104


Code facilitating the generation of a RNN-GAN
"""

import tensorflow as tf 

from tensorflow.keras import optimizers 

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

import numpy as np 
from scipy import stats
import random
import argparse
import itertools

import os

import yaml
from yaml import Loader

from keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences 

from nn_architecture import rnn_gan 
from train import load_data 


# Load yaml file

stream = open(f"configs/trainGAN.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

####################################

# FROM CONFIGURATION FILE, LOAD UP HYPERPARAMETERS 

# Model hyperparameters 

neurons = cnfg['neurons']
num_layers = cnfg['hidLayers']
model_name = cnfg['model_name']
model_path = f'models/{model_name}'

# Load up how we wanna split up our data
direc_subfolder = cnfg['direc_name']
direc = f'Batched Data/{direc_subfolder}'
window_size = cnfg['window_size'] 
series_length = cnfg['series_length']
forecast_length = cnfg['forecast']
full_time_series = True
input_list = cnfg['input_list']
num_features = len(input_list)
num_inputs = len(input_list)

# Load up training params
epochs = cnfg['epochs']
init_lr = cnfg['init_lr']
patience = cnfg['patience'] # For how many epochs do we wait before we start adjusting the LR 
lr_reduce_factor = cnfg['lr_reduce_factor'] # By how much do we update the LR if we trigger reduce_lr?
batch_size = cnfg['batch_size'] # size of batch 
trainTest_split = cnfg['trainTest_split'] # Split between data seen during training and unseen data for testing 
trainVal_split = cnfg['trainVal_split'] # Split between training data and validation data

sizeOfFiles = len([name for name in os.listdir(f'{direc}/.')]) # Global parameter
print(f"Number of files:{int(sizeOfFiles)}")
num_examples_train = int(sizeOfFiles*trainTest_split)

###############################

# Let's load the data 

X,y = load_data(direc, series_length, input_list, window_size, full_time_series, forecast_length)
X_train, y_train = X[0:num_examples_train], y[0:num_examples_train]

# This time, we load the input training data as a dataset 

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(num_examples_train).batch(batch_size)

# Here goes nothing? Let's initialize the generative and discriminator optims. They shouldn't overtake one another

gen_optim = optimizers.AdamW(learning_rate=init_lr, weight_decay=0.001)
dis_optim =  optimizers.AdamW(learning_rate=init_lr, weight_decay=0.001)

# Instantiate our object 

model = rnn_gan(neurons,num_layers, num_inputs, model_name, num_features, forecast_length, model_path)
model.train(train_dataset, epochs, gen_optim, dis_optim)























