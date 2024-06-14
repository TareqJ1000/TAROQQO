
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
        
        
        # Save loss as a csv file for future reference 
        
        complete_loss = np.array([self.metrics['loss'], self.metrics['val_loss']])
        complete_loss = pd.DataFrame(complete_loss)
        
        #complete_loss = pd.DataFrame(hist.history)
        with open(f"loss/loss_{model_name}.csv", "wb") as f:
            complete_loss.to_csv(f)
        
        print('Saved plot of most recent training epoch to disk')
        
# Computes the mean squared error between predicted and true time series
# A small epsilon terms prevents the network from diverging suddenly due to inifinitely small MSE. Otherwise, the MSE is computed exactly the same way. 

def mse_mod(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    loss_true = tf.reduce_mean(loss)
    
    # We add a small epsillion to the MSE. This makes it so that we avoid crazy losses
    return loss_true + 1e-8


# This is a routine that computes every possible permutation of input features. 
# This was used extensively for our preliminary grid search; it is now depreceated 

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

# Routine to normalize between 0 and 1. Any values that are above 1 are zero'd out. 
def norm_data(x):
    minX = np.min(x[np.nonzero(x)])
    maxX = np.max(x[np.nonzero(x)])
    normed = (x - minX)/(maxX - minX)
    
    # Zero out any values that are above 1
    normed[normed>1] =  0
    
    return normed, minX, maxX

# This normalizes the data given that the normalization is determined externally w/ the above routine and e.g. using another dataset. 
# x -- list -- column to be normalized 
# minX - double -- minimum reported value of x. This is calculated externally 
# maxX - double -- maximum reported value of x. This is calculated externally

def norm_data_select(x, minX, maxX):  
    normed = (x - minX)/(maxX - minX)
    # Usually, we would zero out values that are not contained within our range. This is no longer a guarentee w/ unseen data. 
    # normed[normed>1] =  0
    
    return normed, minX, maxX

# Reads hour entries in the .csv files and converts to pythonic integer.
# This is specific to the format of our data

def hours_to_int(x):
    if x[0:-2]=='':
        return 0
    else:
        return int(x[0:-2])
    
# Converts time in a hour:minute format into seconds
# hour - int - hours from 0 to 23
# minute - int - minutes from 0 to 59
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

### ARCHITECTURE HYPERPARAMETERS ###

model_name = cnfg['model_name']
model_name += f"_{seed}"
model_path = f'models/{model_name}' 
nn_type = cnfg['nn_type'] # Architecture type 
neurons = cnfg['neurons'] # Number of neurons/units per layer 
hidLayers = cnfg['hidLayers'] # Number of hidden layers 
window_size = cnfg['window_size'] # This process is the identity if it is set to 1
series_length = cnfg['series_length'] # Input series length 
forecast_length = cnfg['forecast'] # Forecast series length

### TRAINING HYPERPARAMETERS ###

trainTest_split = cnfg['trainTest_split'] # Split between data seen during training and unseen data for testing 
trainVal_split = cnfg['trainVal_split'] # Split between training data and validation data
epochs = cnfg['epochs'] # maximum number of training epochs 
init_lr = cnfg['init_lr'] # starting learning rate 
patience = cnfg['patience'] # For how many epochs do we wait before we start adjusting the LR 
lr_reduce_factor = cnfg['lr_reduce_factor'] # By how much do we update the LR if we trigger reduce_lr?
batch_size = cnfg['batch_size'] 

### DATA HYPERPARAMETERS ### 

# Load up how we wanna split up our data
direc_subfolder = cnfg['direc_name']
direc = f'{direc_subfolder}'
full_time_series = cnfg['full_time_series'] # do we forecast a complete time series, or just the last time step? 
num_of_examples = eval(cnfg['num_of_examples']) # total number of datapoints to load 
num_of_examples_fixed = eval(cnfg['num_of_examples_fixed']) # Fixed number of examples that we wanna load. 
num_of_examples_fixed = int(num_of_examples_fixed)
time_res = cnfg['time_res'] # Time resolution of output network. 
patience = cnfg['patience'] # this determines the amount of epochs we let the network 'stagnate' before stopping training 
shuffler = cnfg['isShuffle'] # do we enable dataset shuffling? 
saveShuffle = cnfg['saveShuffle'] # save the randomized integers. 
loadShuffle = cnfg['loadShuffle'] # do we load the randomized integers?
shuffleDirec = cnfg['shuffleDirec'] + '.txt' # Where are the saved shuffled integers? Only relevant if we choose to load the shuffled ints. 
zeroInput = cnfg['zeroInput'] # Do we zero out the inputs? This is one way of telling us the baseline performance of our model. 

# Select subset of features that we'd like to use with the network. The feature that we select is dependent on the slurm index. 

feature_subset = cnfg['input_list'] # desired input feature-list 

number_of_features = len(feature_subset)
# For temporal features, we include the sin and cos component of that feature. 
if ('day' in feature_subset):
    number_of_features+= 1
if ('time' in feature_subset):
    number_of_features+= 1
    
# Compute the length of the output array factoring in the desired forecast length and the time resolution 

output_len = int(forecast_length/time_res) 


# Loads data in preparation for training. 
#
# direc_name -- string -- name of directory folder where data is located
# time_steps -- int -- number of INPUT time steps
# input_list -- list (of strings) -- names of input features
# num_examples -- int -- maximum number of examples to load
# full_time_series -- boolean -- do we output the full time series or just the last time step. 
# forecast_len -- int -- by how much ahead are we making the forecast 
# time_res -- int -- time resolution of output steps 
# zeroInput -- boolean -- if enabled, the input data is completely zero'd out. Use only for analytical purposes to determine how clever your actual network is. 

def load_data(direc_name, time_steps, input_list, num_of_examples, full_time_series=False, forecast_len=1, time_res=1, zeroInput = False):

    total_input = []
    total_output = []
    
    # Files expected
    
    directory_list = [name for name in os.listdir(f'{direc_name}/.')]
    num_features = len(input_list)
    if ('day' in input_list):
        num_features+= 1
    if ('time' in input_list):
        num_features+= 1
    
    print(f'Parameter List: {input_list}')
    
    for jj, name in enumerate(directory_list):
        df = pd.read_csv(f'{direc_name}/{name}')
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

        # Instantiate the input and output dataset
        
        dataset_weather = np.empty((time_steps, num_features))
        dataset_output = np.empty((output_len, 1))
        
        ###### INPUT DATA #######
        
        ii = 0
        kk = 0 # index for column
        
        while ii < num_features:
            
           if(zeroInput): # If zero input is enabled, we are zeroing out all of the inputs. 
               dataset_weather[:,ii] = np.zeros(time_steps)
               ii += 1
           else: # Access from the dataset the 'colName' feature
                colName = input_list[kk]
                
                if(colName=='day'): # include both day_sin and day_cos
                
                    dataset_weather[:,ii] = df['day_sin'].to_numpy()
                    ii += 1
                    dataset_weather[:,ii] = df['day_cos'].to_numpy()
                    
                elif(colName=='time'): # Include both time_sin and time_cos
                
                    dataset_weather[:,ii] = df['time_sin'].to_numpy()
                    ii += 1
                    dataset_weather[:,ii] = df['time_cos'].to_numpy()
                   
                elif(colName=='CN2'): # We are interested in the log_10 value of input/output CN2
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
        
        total_input.append(dataset_weather)
        total_output.append(dataset_output)
            
        if (jj%100==0):
            print(f"Data loaded:{jj}")
            
        if (jj>num_of_examples): # Halt loading data if we have exceeded stated number of examples
            print("Finished loading data!")
            break;
            
    total_input = np.array(total_input)
    total_output = np.array(total_output)
    
    if(full_time_series): # Return the full time series, or only the last time step in the forecasted time series. 
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
    
    # We can begin proper. Load up the dataset and get ready to train. 
    X,y = load_data(direc, series_length, feature_subset, num_of_examples, full_time_series, forecast_length, time_res, zeroInput=zeroInput)
    num_of_examples = len(X)

    # If shuffling is enabled, apply shuffling of the dataset 

    if shuffler:
        shuffleInts = np.arange(0,num_of_examples)
        if (loadShuffle):
            print("Loading shuffled integers")
            oldShuffle = np.loadtxt(shuffleDirec, dtype='int')
            # Now, we be careful about the length here 
            old_num_of_examples = len(oldShuffle)
            if(old_num_of_examples == num_of_examples or num_of_examples > old_num_of_examples):
                X = X[oldShuffle]
                y = y[oldShuffle]
                # The second case needs to be handled with more care becuase we might have integers that are out of bounds wrt the currently loaded system. 
            elif(old_num_of_examples > num_of_examples):
                print("The shuffled integers is greater than the loaded number of examples. ")
                for ii in range(num_of_examples):
                    if(oldShuffle[ii] >= num_of_examples): # skip the shuffle and move on to the next iteration. 
                        continue
                    else:
                        temp = X[ii]
                        X[ii] = X[oldShuffle[ii]]
                        X[oldShuffle[ii]] = temp
        else:
            np.random.shuffle(shuffleInts)
            if (saveShuffle):
                np.savetxt(f'shuffleInts/{model_name}.txt', shuffleInts)
            X = X[shuffleInts]
            y = y[shuffleInts]
        
    # The shuffling step will be the same as before (just w/ more data). We select the first number of examples examples from that shuffled integer list. 
    
    if(num_of_examples_fixed>0):
        X = X[0:num_of_examples_fixed]
        y = y[0:num_of_examples_fixed]
    
    # Then split the data set into train-val-test
    num_examples_train = int(len(X)*trainTest_split)
    num_examples_val = int(len(X)*trainVal_split)
    num_trainVal = num_examples_train + num_examples_val 
    num_examples_test = num_of_examples - num_trainVal

    # Export the test data for post-analysis. Allow for atleast 1000 data samples in the final dataset. 
    X_test, y_test = X[num_trainVal:], y[num_trainVal:]

    print(f"Number of training examples: {num_examples_train}")
    print(f"Number of validation examples: {num_examples_val}")
    print(f"Number of test examples: {num_examples_test}")
    
    total_len_train = num_examples_train
    # The non-test data is partitioned into training and validation data for the network. 
    X_train, y_train = X[0:num_examples_train], y[0:num_examples_train]
    X_val, y_val = X[num_examples_train:num_trainVal], y[num_examples_train:num_trainVal]
    
    
    if (zeroInput): # The input data is all zeros, so it only makes sense to normalize the output data. 
    
        y_train[:,:,0], minCN2, maxCN2 = norm_data(y_train[:,:,0])
        y_val[:,:,0], _, _ = norm_data_select(y_val[:,:,0], minCN2, maxCN2)
        y_test[:, :, 0], _, _ = norm_data_select(y_test[:,:,0], minCN2, maxCN2)
        print(f"Activated ZERO mode. Normalization of OUTPUT training data finished! Minimums: {minCN2}, Maximums: {maxCN2}. This is applied seperately to validation & test data.")
        
    else:
        # normalize the training data. The convention is that the last element in the minTrain/maxTrain list is always the CN2 value
        
        minTrain = []
        maxTrain = []
        
        for ii in np.arange(number_of_features  - 1):
            X_train[:,:,ii], minOut, maxOut = norm_data(X_train[:,:,ii])
            minTrain.append(minOut)
            maxTrain.append(maxOut)
        
        # The CN2 input and output should be normalized together. 
        
        cn2_combined_train = np.concatenate((X_train[:,:,-1].flatten(), y_train[:,:,0].flatten()))
        _, minCN2, maxCN2 = norm_data(cn2_combined_train)
        minTrain.append(minCN2)
        maxTrain.append(maxCN2)
        
        # Update X_train, y_train
        
        X_train[:,:,-1], _, _ = norm_data_select(X_train[:,:,-1], minCN2, maxCN2)
        y_train[:,:,0], _, _  = norm_data_select(y_train[:,:,0], minCN2, maxCN2)
        
        # Repeat this procedure on the validation and test datasets, but now utilize the normalization of the training data.
        # It should be okay to normalize the CN2 here as well as the max and min are fixed. 
        
        for ii in np.arange(number_of_features):
            X_val[:,:,ii], _, _= norm_data_select(X_val[:,:,ii], minTrain[ii], maxTrain[ii])
            X_test[:,:,ii],_,_ = norm_data_select(X_test[:,:,ii], minTrain[ii], maxTrain[ii])
            
            if (ii==number_of_features-1):
                y_val[:,:,0],_,_ = norm_data_select(y_val[:,:,0], minTrain[ii], maxTrain[ii])
                y_test[:,:,0],_,_ = norm_data_select(y_test[:,:,0], minTrain[ii], maxTrain[ii])
      
        print(f"Normalization of training data finished! Minimums: {minTrain}, Maximums: {maxTrain}. This is applied seperately to validation & test data.")
    
    # For one type of FFNN, we flatten out the input distribution.   
    
    if(nn_type==1):
        X_train_shape = np.shape(X_train)
        X_val_shape = np.shape(X_val)
        X_test_shape = np.shape(X_test)
        
        X_train = np.reshape(X_train,(X_train_shape[0], X_train_shape[1]*X_train_shape[2]))
        X_val = np.reshape(X_val, (X_val_shape[0], X_val_shape[1]*X_val_shape[2]))
        X_test = np.reshape(X_test, (X_test_shape[0], X_test_shape[1]*X_test_shape[2]))
    
    # We should now be ready to roll. Save the test data for future analysis! 
    
    with open(f"Test Data/{model_name}_testData.pkl", 'wb') as f:
        pkl.dump([X_test, y_test], f)
        print(f"Test data saved as {model_name}_testData.pkl ")
    
    model = rn_network(nn_type, neurons, 1, number_of_features, hidLayers, model_name, forecast_len=output_len)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False, verbose=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = lr_reduce_factor, patience = patience, min_lr = 1e-7)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, start_from_epoch=100)
    
    # Compile and run the model 

    adam_optimizer=optimizers.Adam(learning_rate=init_lr)
    
    model.mynn.compile(loss=mse_mod, optimizer=adam_optimizer)
    hist = model.mynn.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs=epochs, callbacks = [PlotLearning(), cp_callback, reduce_lr, early_stop], verbose=2)
    
    # Save loss as a csv file for future reference 
    
    complete_loss = pd.DataFrame(hist.history)

    with open(f"loss/loss_{model_name}.csv", "wb") as f:
        complete_loss.to_csv(f)
        
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
















