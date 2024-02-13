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

import os

import yaml
from yaml import Loader


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

def norm_data(x):
    for ii in range(len(x)):
        x[ii] = (x[ii] - np.min(x))/(np.max(x) - np.min(x))
    return x


def load_data(direc_name, time_steps):

    total_input = []
    total_output = []
    
    # Files expected
    
    directory_list = [name for name in os.listdir(f'{direc_name}/.')]
    
    
    for ii, name in enumerate(directory_list):
        df = pd.read_csv(f'{direc_name}/{name}')
        
        dataset_weather = np.empty((time_steps, 4))
        dataset_output = np.empty((1, 1))
        
        ###### INPUT DATA #######
        
        # In the 0th input, temperature
        dataset_weather[:,0] = df["temperature"].to_numpy()
        # In the 1st input, relative humidity (%)
        dataset_weather[:,1] = df["relative_humidity"].to_numpy()
        # In the 2nd input, standard pressure 
        dataset_weather[:,2] = df["pressure_station"].to_numpy()
        #In the 3rd input, Solar Radiation 
        dataset_weather[:,3] = df["solar_radiation"].to_numpy()
        total_input.append(dataset_weather)
        
        ###### OUTPUT DATA #######
        
        # In the 0th output, CN2 FUTURE
        dataset_output[:,0] = np.log10(eval(df["CN2-R0 Future"][1]))
        # In the 1st output, R0
        # dataset_output[:,1] = df["CN2-R0 Future"][2]  
    
        total_output.append(dataset_output)
        
    total_input = np.array(total_input)
    total_output = np.array(total_output)
    
    # Apply normalization to each input entry
    
    total_input[:,:,0] = norm_data(total_input[:,:,0])
    total_input[:,:,1] = norm_data(total_input[:,:,1])
    total_input[:,:,2] = norm_data(total_input[:,:,2])
    total_input[:,:,3] = norm_data(total_input[:,:,3])
    
    total_output[:,:,0] = norm_data(total_output[:,:,0])
   # total_output[:,:,1] = norm_data(total_output[:,:,1])
    
    return total_input, total_output


if __name__ == '__main__':
    
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
    model_path = f'models/{model_name}'

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


    # Compute total number of samples contained in the subfolder. This'll let us calculate the number of examples that will be used for the training 
    sizeOfFiles = len([name for name in os.listdir(f'{direc}/.')]) # Global parameter
    print(f"Number of files:{int(sizeOfFiles)}")
    num_examples_train = int(sizeOfFiles*trainTest_split)

    # Note this number down!! 
    print(f"Number of training examples: {num_examples_train}")
    
    # We can begin proper. Load up the dataset and get ready to train!!
    
    X,y = load_data(direc,12)
    X_train, y_train = X[0:num_examples_train], y[0:num_examples_train]
    
    model = rn_network(nn_type, neurons, 1, 4, hidLayers, model_name)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_reduce_factor, patience = patience, min_lr = 1e-7, verbose = 1)
    
    # Compile and run the model 
    
    adam_optimizer=optimizers.Adam(learning_rate=init_lr)
    model.mynn.compile(loss='mse', optimizer=adam_optimizer)
    hist = model.mynn.fit(X_train, y_train, batch_size=batch_size, validation_split=trainVal_split, epochs=epochs, callbacks = [PlotLearning(), cp_callback, reduce_lr], verbose=2)
    
    # Save loss as a csv file for future reference 
    
    complete_loss = pd.DataFrame(hist.history)

    with open(f"loss/loss_{model_name}.csv", "wb") as f:
        complete_loss.to_csv(f)
        
    
        
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

















