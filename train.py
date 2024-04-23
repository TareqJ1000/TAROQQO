
# -*- coding: utf-8 -*-
"""
Code which initializes the training
"""
from nn_architecture import rn_network

import tensorflow as tf

from tensorflow.keras import optimizers 

import pandas as pd

import matplotlib.pyplot as plt
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
import pickle as pkl

# from augmentation import jitter, scaling, magnitude_warp, window_slice, window_warp

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
        
        
def mse_mod(y_true, y_pred):

    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    loss_true = tf.reduce_mean(loss)
    
    # We add a small epsillion to the MSE. This makes it so that we avoid crazy losses
    return loss_true + 1e-8

def get_column_subsets():
    column_names = ['pressure', 'temperature', 'wind_speed', 'SOG', 'day', 'time']
    # Create all possible permutations of this list
    subsets = [list(itertools.combinations(column_names, ii)) for ii in range(len(column_names)+1)]
    base_names = ['solar_radiation', 'relative_humidity']
    all_subsets = []
    
    num_of_combo = np.arange(len(column_names)+1)
    
    for ii in num_of_combo:
        for jj in range(len(subsets[ii])):
            temp = np.concatenate((base_names, subsets[ii][jj]))
            temp = np.concatenate((temp, np.array(['CN2']))) # CN2 should always be the last thing. 
            all_subsets.append(temp)
    
    return all_subsets

# Routine to normalize between 0 and 1 (Following the atmospheric turbulence paper)

def norm_data(x):
    minX = np.min(x[np.nonzero(x)])
    maxX = np.max(x[np.nonzero(x)])
    normed = (x - minX)/(maxX - minX)
    
    # Zero out any values that are above 1
    normed[normed>1] =  0
    
    return normed, minX, maxX


# This normalizes the data according to prior data. 

def norm_data_select(x, minX, maxX):  
    normed = (x - minX)/(maxX - minX)
    
    # Usually, we would zero out values that are not contained within our range. This is no longer a guarentee w/ unseen data. 
    # normed[normed>1] =  0
    
    return normed, minX, maxX

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

def hours_to_int(x):
    if x[0:-2]=='':
        return 0
    else:
        return int(x[0:-2])
    
def convert_to_sec(minute, hour):
    return 3600*hour + 60*minute

# parse through slurm array (for use w/ bash script. You can set shift to be a random integer for the purposes of testing locally)

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

# OMIT IN CLUSTER 

shift = 0

# OMIT IN CLUSTER

# Loads up and prepares the data for training

stream = open(f"configs/train{shift}.yaml", 'r')
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
diff = cnfg['diff'] # Setting this option to -1 is the same as not starting it altogether 
# Load up how we wanna split up our data
direc_subfolder = cnfg['direc_name']
direc = f'{direc_subfolder}'
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
feature_subset = feature_subsets[-1]
#feature_subset = ['RH %', 'kJ/m^2', 'CN2', 'Temp Â°C']
#feature_subset = cnfg['input_list']

number_of_features = len(feature_subset)
if ('day' in feature_subset):
    number_of_features+= 1
if ('time' in feature_subset):
    number_of_features+= 1
    
full_time_series = cnfg['full_time_series']
model_path = f'models/{model_name}'
num_of_examples = eval(cnfg['num_of_examples'])
time_res = cnfg['time_res'] # Time resolution of output network. 
pad_output_zeros=cnfg['pad_output_zeros']
patience = cnfg['patience']


# Compute the length of the output array factoring in the desired forecast length and the time resolution 
output_len = int(forecast_length/time_res) 

def load_data(direc_name, time_steps, input_list, window_size, num_of_examples, full_time_series=False, pad_output_zeros = True,  forecast_len=1, time_res=1):

    total_input = []
    total_output = []
    
    # Files expected
    
    directory_list = [name for name in os.listdir(f'{direc_name}/.')]
    num_features = len(input_list)
    print(num_features)
    if ('day' in input_list):
        num_features+= 1
    if ('time' in input_list):
        num_features+= 1
    print(num_features)
    
    
    print(f'Parameter List: {input_list}')
    
    num_of_zeros = 0
    
    for jj, name in enumerate(directory_list):
        
        df = pd.read_csv(f'{direc_name}/{name}')
        #print(name)
        # rename columns to something more decipherable 
        df = df.rename(columns={'Temp °C':'temperature', 'RH %':'relative_humidity', 'kJ/m^2':'solar_radiation', 'Wind Speed m/s':'wind_speed', 'SOG cm':'SOG','Pressure hPa':'pressure', 'hr:min (UTC)':'time', 'Julian day (UTC)': 'day'})
        # Map the day into a unit circle, and create 'day_sin'  and 'day_cos' to define the x and y components in the circle. 
        df['day_sin'] = np.sin(df['day']*(2.*np.pi/365))
        df['day_cos'] = np.cos(df['day']*(2.*np.pi/365))
        # For time, we convert to string representation
        df['time']=df['time'].astype(str)
        df['minute'] = df['time'].apply(lambda x: int(x[-2:]))
        df['hour'] = df['time'].apply(lambda x: hours_to_int(x))
        df['second'] = convert_to_sec(df['minute'], df['hour'])
        # Map the time into a unit circle 
        df['time_sin']= np.sin(df['second']*(2.*np.pi/86400))
        df['time_cos']= np.cos(df['second']*(2.*np.pi/86400))

        #print(df.columns)
        #input()
        
        # If the prior/future CN2 columns have zero values, then continue to next iteration 
        if(df['CN2']==0).any() or (df['CN2 Future']==0).any():
             #print(jj)
             num_of_zeros += 1
             print(f'number of zeros: {num_of_zeros}')
             print('error data detected. Skipping to next value')
             continue
        
        dataset_weather = np.empty((time_steps, num_features))
        dataset_output = np.empty((output_len, 1))
        
        ###### INPUT DATA #######
        ii= 0
        kk= 0 # index for column
        
        while ii < num_features:
           # print(ii)
            colName = input_list[kk]
            if(colName=='day'):
             
                # include both day_sin and day_cos
                
                dataset_weather[:,ii] = df['day_sin'].to_numpy()
                ii += 1
                dataset_weather[:,ii] = df['day_cos'].to_numpy()
                
            elif(colName=='time'):
                
                # Include both time_sin and time_cos
                
                dataset_weather[:,ii] = df['time_sin'].to_numpy()
                ii += 1
                dataset_weather[:,ii] = df['time_cos'].to_numpy()
               
                
            
            elif(colName=='CN2'):
                dataset_weather[:,ii] = np.log10(df[colName].to_numpy())
            else:
                
                dataset_weather[:,ii] = df[colName].to_numpy()
            kk += 1
            ii += 1
                
        ###### OUTPUT DATA #######
        
        # In the 0th output, CN2 FUTURE
        
        # First, let's consider every example up to forecast length 
        nn_output  = np.log10(df["CN2 Future"][:forecast_len].to_numpy())
        
        # Next, only consider every time_res example in the final output
        dataset_output[:,0] = nn_output[np.mod(np.arange(len(nn_output)),time_res) == 0]
        
        # Let's consider wildly varying output data. Compute the difference between maximum and minimum. 
        max_CN2 = np.max(np.abs(dataset_output[:,0]))
        min_CN2 = np.min(np.abs(dataset_output[:,0]))
        diff = np.abs(max_CN2 - min_CN2)
        
    
        
        if (diff >= cnfg['diff']):
            total_input.append(dataset_weather)
            total_output.append(dataset_output)
            
        if (jj%500==0):
            print(f"Data loaded:{jj}")
            
        if (jj>num_of_examples):
            print("Finished loading data!")
            break;
                
    total_input = np.array(total_input)
    total_output = np.array(total_output)
    
    if(full_time_series):
        return total_input, total_output
    else:
        return total_input, total_output[:,0]
    

    
if __name__ == '__main__':

    print(f'Parameters used: {feature_subset}. Saving as txt...')
    with open(f'params/{model_name}.txt','w') as txt_file:
        txt_file.write(str(feature_subset))
    
    # Compute total number of samples contained in the subfolder. This'll let us calculate the number of examples that will be used for the training 
    
    sizeOfFiles = len([name for name in os.listdir(f'{direc}/.')]) # Global parameter
    print(f"Number of files:{int(sizeOfFiles)}")
    
    # We can begin proper. Load up the dataset and get ready to train!!
    
    X,y = load_data(direc, series_length, feature_subset, window_size, num_of_examples, full_time_series, pad_output_zeros, forecast_length, time_res)
    num_of_examples = len(X)

    # Then split the data set into train-val-test
    
    num_examples_train = int(len(X)*trainTest_split)
    X_data, y_data= X[0:num_examples_train], y[0:num_examples_train]
    
    # Export the test data for post-analysis. Allow for atleast 1000 data samples in the final dataset. 
    X_test, y_test = X[num_examples_train:], y[num_examples_train:]

    print(f"Number of training+validation examples: {num_examples_train}")
    
    total_len_train = len(X_data)
    num_examples_val = int(total_len_train*(1-trainVal_split))
    
    X_train, y_train = X_data[0:num_examples_val], y_data[0:num_examples_val]
    X_val, y_val = X_data[num_examples_val:], y_data[num_examples_val:]
    
    # normalize the training data. The convention is that the last element in the minTrain/maxTrain list is always the CN2 value
    
    minTrain = []
    maxTrain = []
    
    for ii in np.arange(number_of_features  - 1):
        X_train[:,:,ii], minOut, maxOut = norm_data(X_train[:,:,ii])
        minTrain.append(minOut)
        maxTrain.append(maxOut)
    
    # The CN2 Input and Output should be normalized together. 
    
    cn2_combined_train = np.concatenate((X_train[:,:,-1].flatten(), y_train[:,:,0].flatten()))
    _, minCN2, maxCN2 = norm_data(cn2_combined_train)
    minTrain.append(minCN2)
    maxTrain.append(maxCN2)
    
    # Update X_train, y_train
    
    X_train[:,:,-1], _, _ = norm_data_select(X_train[:,:,-1], minCN2, maxCN2)
    y_train[:,:,0], _, _  = norm_data_select(y_train[:,:,0], minCN2, maxCN2)
    
    # Repeat this procedure on the validation and test datasets, but now utilize the normalization of the training data. It should be okay to normalize the CN2 here as well as the max and min are fixed. 
    
    for ii in np.arange(number_of_features):
        X_val[:,:,ii], _, _= norm_data_select(X_val[:,:,ii], minTrain[ii], maxTrain[ii])
        X_test[:,:,ii],_,_ = norm_data_select(X_test[:,:,ii], minTrain[ii], maxTrain[ii])
        
        if (ii==number_of_features-1):
            y_val[:,:,0],_,_ = norm_data_select(y_val[:,:,0], minTrain[ii], maxTrain[ii])
            y_test[:,:,0],_,_ = norm_data_select(y_test[:,:,0], minTrain[ii], maxTrain[ii])
            
            
    print(f"Normalization of training data finished! Minimums: {minTrain}, Maximums: {maxTrain}. This is applied seperately to validation & test data.")
    
    # We should now be ready to roll. Save the test data and let's get started! 
    
    with open(f"Test Data/{model_name}_testData.pkl", 'wb') as f:
        pkl.dump([X_test, y_test], f)
        print(f"Test data saved as {model_name}_testData.pkl ")
    
    print(f"Applying {augument_technique} augumentation technqiue... ")
    
    if augument_technique=='jitter':
        #X_aug = jitter(X_train, sigma=cnfg['jitter_sigma'])
        y_aug = y_train
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    if augument_technique=='window_warp':
        #X_aug = window_warp(X_train, window_ratio=cnfg['windowWarp_ratio'])
        y_aug = y_train 
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    if augument_technique=='window_slice':
        #X_aug = window_slice(X_train, reduce_ratio=cnfg['reduce_ratio'])
        y_aug = y_train
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
    if augument_technique=='magnitude_warp':
        #X_aug = magnitude_warp(X_train, sigma=cnfg['magnitude_sigma'], knot=cnfg['magnitude_knot'])
        y_aug = y_train 
        
        X_train = np.concatenate((X_train, X_aug))
        y_train = np.concatenate((y_train, y_aug))
        
        
    model = rn_network(nn_type, neurons, 1, number_of_features, hidLayers, model_name, forecast_len=output_len)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, verbose=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_reduce_factor, patience = patience, min_lr = 1e-7)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, start_from_epoch=100)
    
    # Compile and run the model 
    
    #adam_optimizer=optimizers.AdamW(learning_rate=init_lr, weight_decay=0.001)
    adam_optimizer=optimizers.Adam(learning_rate=init_lr)
    
    model.mynn.compile(loss=mse_mod, optimizer=adam_optimizer)
    hist = model.mynn.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs=epochs, callbacks = [PlotLearning(), cp_callback, reduce_lr, early_stop], verbose = 2)
    
    # Save loss as a csv file for future reference 
    
    complete_loss = pd.DataFrame(hist.history)

    with open(f"loss/loss_{model_name}.csv", "wb") as f:
        complete_loss.to_csv(f)
        
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
















