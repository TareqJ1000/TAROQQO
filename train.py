
# -*- coding: utf-8 -*-
"""
Code which initializes the training
"""
from nn_architecture import rn_network

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

from augmentation import jitter, scaling, magnitude_warp, window_slice, window_warp

# Class used to plot each epoch of the network. This is so that we get live feedback on how well our network is learning at each epoch. Credit to this medium article (https://medium.com/geekculture/how-to-plot-model-loss-while-training-in-tensorflow-9fa1a1875a5_) and to the QPT paper from before

class PlotLearning(tf.keras.callbacks.Callback):
    
    """
    Callback to plot the learning curves of the model during training.
    """
    
    def __init__(self):
        super(PlotLearning,self).__init__()
    
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        print(' Saving current plot of training evolution')
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        # Plotting
        # metrics = [x for x in logs if 'val' not in x]
            
        f, axs = plt.subplots(1, 1, figsize=(5,5))
        clear_output(wait=True)

        axs.plot(range(1, epoch + 2), 
                self.metrics['loss'], 
                label='loss')
        axs.plot(range(1, epoch + 2), 
                self.metrics['val_loss'], 
                label='val_loss')

        axs.legend()
        axs.grid()
            
        model_name = self.model.name
        directory = f'plots/{model_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.tight_layout()
        plt.savefig(directory + f'/epochs_{epoch}.png')
        
        print('Saved plot of most recent training epoch to disk')
        
        
def mse_trans(y_true, y_pred):

    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    loss_true = tf.reduce_mean(loss)
    
    return loss_true
        

def get_column_subsets():
    column_names = ['pressure_station', 'pressure_sea','dew_point','temperature', 'cloud_cover_8']
    # Create all possible permutations of this list
    subsets = [list(itertools.combinations(column_names, ii)) for ii in range(len(column_names)+1)]
    base_names = ['solar_radiation', 'relative_humidity', 'CN2']
    all_subsets = []
    
    num_of_combo = np.arange(5+1)
    
    for ii in num_of_combo:
        for jj in range(len(subsets[ii])):
            temp = np.concatenate((base_names, subsets[ii][jj]))
            all_subsets.append(temp)
    
    return all_subsets

def norm_data(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

# This applies a rolling average on the dataset 

# Function taken from a learnpython article

def roll_average(input_data, window_size):
    result = []
    for i in range(len(input_data) - window_size + 1):
        window = input_data[i:i+window_size]
        window_average = sum(window)/window_size
        result.append(window_average)
        
    return np.array(result)
    
def rollify_training(X, window_size):
    
    X_features = X.shape[2]
    
    X_roll_len = X.shape[1] - window_size + 1
    
    X_roll = np.empty((len(X), X_roll_len, X_features))
    
    for ii in range(len(X)):
        for jj in range(X_features):
            X_roll[ii,:,jj] = roll_average(X[ii,:,jj], window_size)
            
    return X_roll


def load_data(direc_name, time_steps, input_list, window_size, full_time_series=False, forecast_len=1):

    total_input = []
    total_output = []
    
    # Files expected
    
    directory_list = [name for name in os.listdir(f'{direc_name}/.')]
    num_features = len(input_list)
    
    print(f'Parameter List: {input_list}')
    
    
    for ii, name in enumerate(directory_list):

        df = pd.read_csv(f'{direc_name}/{name}')
    
        dataset_weather = np.empty((time_steps, num_features))
        dataset_output = np.empty((forecast_len, 1))
        
        ###### INPUT DATA #######
        
        for ii, colName in enumerate(input_list):
            if(colName=='CN2'):
                dataset_weather[:,ii] = np.log10(df[colName].to_numpy())
            else:
                dataset_weather[:,ii] = df[colName].to_numpy()
            
        total_input.append(dataset_weather)
        
        ###### OUTPUT DATA #######
        
        # In the 0th output, CN2 FUTURE
        
        dataset_output[:,0] = np.log10(df["CN2 Future"][:forecast_len].to_numpy())

        total_output.append(dataset_output)
        
    total_input = np.array(total_input)
    total_output = np.array(total_output)
    
    # Apply rolling average onto the input data
    total_input = rollify_training(total_input, window_size)
    
    
    # Apply normalization to each input entry
    for ii in range(num_features):
        total_input[:,:,ii] = norm_data(total_input[:,:,ii])
        
        
    # If we are working with time series prediction, apply rolling on output time series
    # UPDATE: Maybe this isn't the best idea, if we think of rolling average as a regularization technique
    
    if (full_time_series):
        total_output = rollify_training(total_output, window_size)
        
    # Apply normalization to each output entry
    total_output[:,:,0] = norm_data(total_output[:,:,0])
    
    
    # Finally, if we're working with a time series, let's pad out the output array with 0s
    
    if (full_time_series):
        window_time_steps_input = total_input.shape[1]
        window_time_steps_output = total_output.shape[1]
        
        total_output_padded = np.zeros((total_input.shape[0], window_time_steps_input, 1))
        total_output_padded[:, :window_time_steps_output, 0] = total_output[:,:,0]
        
        return total_input, total_output_padded
    
    else:
        return total_input, total_output

if __name__ == '__main__':

    # parse through slurm array (for use w/ bash script. You can set shift to be a random integer for the purposes of testing locally)

    parser=argparse.ArgumentParser(description='test')
    parser.add_argument('--ii', dest='ii', type=int,
        default=None, help='')
    args = parser.parse_args()
    shift = args.ii
    
    # Loads up and prepares the data for training

    stream = open(f"configs/train.yaml", 'r')
    cnfg = yaml.load(stream, Loader=Loader)

    # Set a random seed for each training instance

    seed = random.randint(1000, 9999)
    print(f'seed: {seed}')
    random.seed(seed)

    # Load up model params
    
    model_name = cnfg['model_name']
    model_name += f"_{seed}"
    nn_type = cnfg['nn_type']
    neurons = cnfg['neurons']
    hidLayers = cnfg['hidLayers']
    window_size = cnfg['window_size'] # This process is the identity if it is set to 1
    series_length = cnfg['series_length']
    forecast_length = cnfg['forecast']
    augument_technique = cnfg['augument_technique']
    # Load up how we wanna split up our data
    direc_subfolder = cnfg['direc_name']
    direc = f'Batched Data/{direc_subfolder}'
    trainTest_split = cnfg['trainTest_split'] # Split between data seen during training and unseen data for testing 
    trainVal_split = cnfg['trainVal_split'] # Split between training data and validation data
    # Load up training params
    epochs = cnfg['epochs']
    init_lr = cnfg['init_lr']
    patience = cnfg['patience'] # For how many epochs do we wait before we start adjusting the LR 
    lr_reduce_factor = cnfg['lr_reduce_factor'] # By how much do we update the LR if we trigger reduce_lr?
    batch_size = cnfg['batch_size']
    # Select subset of features that we'd like to use with the network. The feature that we select is dependent on the slurm index. 
    feature_subsets = get_column_subsets()
    #feature_subset = feature_subsets[shift]
    feature_subset = ['relative_humidity', 'solar_radiation', 'CN2', 'temperature']
    number_of_features = len(feature_subset)
    full_time_series = True
    model_path = f'models/{model_name}'
    
    
    print(f'Parameters used: {feature_subset}. Saving as txt...')
    with open(f'params/{model_name}.txt','w') as txt_file:
        txt_file.write(str(feature_subset))
    
    # Compute total number of samples contained in the subfolder. This'll let us calculate the number of examples that will be used for the training 
    
    sizeOfFiles = len([name for name in os.listdir(f'{direc}/.')]) # Global parameter
    print(f"Number of files:{int(sizeOfFiles)}")
    num_examples_train = int(sizeOfFiles*trainTest_split)

    # Note this number down!! 
    print(f"Number of training+validation examples: {num_examples_train}")
    
    # We can begin proper. Load up the dataset and get ready to train!!
    
    X,y = load_data(direc, series_length, feature_subset, window_size, full_time_series, forecast_length)
    X_data, y_data= X[0:num_examples_train], y[0:num_examples_train]
    
    total_len_train = len(X_data)
    num_examples_val = int(total_len_train*(1-trainVal_split))
    
    # We will be applying regularization seperately 
    
    X_train, y_train = X_data[0:num_examples_val], y_data[0:num_examples_val]
    X_val, y_val = X_data[num_examples_val:], y_data[num_examples_val:]
    
    print(f"Training data rolled with window size of {window_size} and normalized! Applying {augument_technique} augumentation technqiue... ")
    
    if augument_technique=='jitter':
        X_aug = jitter(X_train, sigma=cnfg['jitter_sigma'])
        y_aug = y_train
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    if augument_technique=='window_warp':
        X_aug = window_warp(X_train, window_ratio=cnfg['windowWarp_ratio'])
        y_aug = y_train 
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    if augument_technique=='window_slice':
        X_aug = window_slice(X_train, reduce_ratio=cnfg['reduce_ratio'])
        y_aug = y_train
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    model = rn_network(nn_type, neurons, 1, number_of_features, hidLayers, model_name, forecast_len=series_length-window_size+1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, verbose=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_reduce_factor, patience = patience, min_lr = 1e-7)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=45, start_from_epoch=100)
    
    # Compile and run the model 
    
    adam_optimizer=optimizers.AdamW(learning_rate=init_lr, weight_decay=0.001)
    model.mynn.compile(loss='mse', optimizer=adam_optimizer)
    hist = model.mynn.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs=epochs, callbacks = [PlotLearning(), cp_callback, reduce_lr, early_stop], verbose=2)
    
    # Save loss as a csv file for future reference 
    
    complete_loss = pd.DataFrame(hist.history)

    with open(f"loss/loss_{model_name}.csv", "wb") as f:
        complete_loss.to_csv(f)
        
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
















