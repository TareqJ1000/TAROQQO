

'''

This code implmenets the recurrent neural network architectures

'''

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,LSTM,GRU, GRUCell
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'
from tensorflow.keras.activations import tanh


class rn_network(tf.Module):
    def __init__(self, nn_type, units, num_features, num_inputs,num_hidden,name,forecast_len=1):
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
        
        
        if (nn_type==3): # GRU-RNN that returns a whole ass time series as output
            conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=True))
            for ii in range(num_hidden):
                conv.add(GRU(units, return_sequences=True))
            conv.add(Dense(num_features, activation='tanh'))
            
        if (nn_type==4): # Single-Shot RNN
            conv.add(GRU(units,input_shape=(None, num_inputs), return_sequences=True))
            for ii in range(num_hidden-1):
                conv.add(GRU(units, return_sequences=True))
            conv.add(GRU(units, return_sequences=False))
            conv.add(Dense(num_features*forecast_len, activation='tanh'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))

        self.mynn = conv
        
    
# Implementing an autoregressive NN (from Tensorflow)

class feedback(tf.keras.Model):
    
    def __init__(self, units, out_steps, num_layers, name, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = GRUCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, input_shape=(None, 4), return_sequences=True)
        self.dense = Dense(num_features)
        
    def warmup(self, inputs):
            x, *state = self.lstm_rnn(inputs)
            pred = self.dense(x)
            return pred, state
    
    def call(self, inputs, training=None):
            predictions= []
            # initialize the GRU state with an input
            prediction, state = self.warmup(inputs) 
            # insert the first prediction 
            predictions.append(predictions)
            # This is a single step time prediction
            
            # now compute the remaining predictions for the rest of time steps
            for n in range(1, self.out_steps):
                x = prediction
                x, state = self.lstm_cell(x, states=state, training=training)
                prediction = self.dense(x)
                predictions.append(prediction)
            
            predictions = tf.stack(prediction) # prediction shape => (time, batch, features)
            predictions = tf.transpose(predictions, [1,0,2]) # prediction => (batch, time, features)
            return predictions

            
        
    
    
    
# riggy = feedback(40, 12, 3, 'goggo', 4)
        
#riggo = rn_network(1,100,2,4,1,'riggy')
#riggo.mynn.summary()

        
#riggo = rn_network(0,100,2,4,20,'riggy')

        
        
            
            
        
            




            
        
