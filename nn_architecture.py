'''

This code implmenets the recurrent neural network architectures

'''

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,LSTM,GRU
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'
from tensorflow.keras.activations import tanh


class rn_network(tf.Module):
    def __init__(self, nn_type, units, num_features, num_inputs,num_hidden,name):
        super(rn_network, self).__init__()
        
        conv = Sequential(name=name)
        
        if (nn_type==0): # GRU-RNN with dense layers as hidden layers 
            conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=False))
            for ii in range(num_hidden):
                conv.add(Dense(int(units/2), activation='tanh'))
            conv.add(Dense(num_features, activation='sigmoid'))
            
        if (nn_type == 1): #GRU-RNN with GRU layers as hidden layers 
            conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=True))
            for ii in range(num_hidden-1):
                conv.add(GRU(units, return_sequences=True))
            conv.add(GRU(units, return_sequences=False))
            conv.add(Dense(num_features, activation='tanh'))
            
            
        if (nn_type == 2): #GRU-RNN with narrower GRU layers as hidden layers 
          conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=True))
          for ii in range(num_hidden-1):
              conv.add(GRU(units/2,input_shape=(None, num_inputs), return_sequences=True))
          conv.add(GRU(units, return_sequences=False))
          conv.add(Dense(32,activation='tanh'))
          conv.add(Dense(num_features, activation='tanh'))
    
        self.mynn = conv
    
        
    
riggo = rn_network(1,100,2,4,1,'riggy')
riggo.mynn.summary()

        
        
            
            
        
            




            
        
