
'''
This code implmenets the recurrent neural network architectures
'''

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,LSTM,GRU, GRUCell, Bidirectional
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'
from tensorflow.keras.activations import tanh

import numpy as np
from IPython import display
import os

import matplotlib.pyplot as plt

import time 

import pandas as pd

# Class for NN model objects
#
# nn_type -- int -- type of NN architecture to be selected 
# units -- int -- number of units/neurons to be featured in each hidden layer
# num_features -- int -- number of output features 
# num_inputs -- int -- number of input features 
# num_hidden -- int -- number of hidden layers
# name -- string -- model name 
# forecast_len -- int -- length of output forecast length 

class rn_network(tf.Module):

    def __init__(self, nn_type, units, num_features, num_inputs,num_hidden,name,forecast_len=1):
        super(rn_network, self).__init__()
        
        conv = Sequential(name=name)
            
        if (nn_type==0): # Single-Shot RNN. This is the flagship architecture featured in our paper
            conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=True))
            for ii in range(num_hidden-1):
                conv.add(GRU(units, return_sequences=True))
            conv.add(GRU(units, return_sequences=False))
            conv.add(Dense(num_features*forecast_len, activation='relu'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))
            
        if (nn_type==1): # This is just a FNN we flatten the input s.t. the first num_inputs entries of the data represent the first time step, the second num_inputs entries of the data represent the second time step, and so on... this is from Grose & Watson
            conv.add(Dense(num_inputs*720, activation='relu'))
            for ii in range(num_hidden-1):
                conv.add(Dense(units, activation='relu'))
            conv.add(Dense(num_features*forecast_len, activation='relu'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))
        
        if (nn_type==2): # Also a FNN but now we only care about the last time step. Taken from the tensorflow tutorial (https://www.tensorflow.org/tutorials/structured_data/time_series)
            conv.add(tf.keras.layers.Lambda(lambda x: x[:, -1:, :]))
            for ii in range(num_hidden-1):
                conv.add(Dense(units, activation='relu'))
            conv.add(Dense(num_features*forecast_len, activation='relu'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))
        
        self.mynn = conv
    

        
        
    
    
    
        
    
        
        
            
            
        
            




            
        
