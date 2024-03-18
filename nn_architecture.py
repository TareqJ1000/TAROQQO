

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


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class rn_network(tf.Module):
    
# Here, we implement direct RNN that return forecasted time series as output

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
            conv.add(Dense(num_features*forecast_len, activation='relu'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))
            
        if (nn_type==5): # RNN with bidirectional layers
            conv.add(Bidirectional(GRU(units,input_shape=(None, num_inputs), return_sequences=True)))
            conv.add(Bidirectional(GRU(units,return_sequences=True)))
            conv.add(Bidirectional(GRU(units)))
            conv.add(Dense(num_features*forecast_len, activation='relu'))
            conv.add(tf.keras.layers.Reshape([forecast_len, num_features]))
                     
        self.mynn = conv
        


# Implementing an animal RNN-GAN. This is a rough approximation of the implementation observed in (https://arxiv.org/abs/1611.09904) 

class rnn_gan(tf.Module):
    
    def __init__(self, units, num_layers, num_inputs,  name, num_features, forecast_len, checkpoint_dir, checkpoint_pre='ckpt'):
        super(rnn_gan, self).__init__()
        self.units = units
        self.num_layers = num_layers
        self.num_features = num_features 
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_pre = checkpoint_pre
        
       # The generator object 
            
        gen_model = Sequential(name=name)
        gen_model.add(GRU(units, input_shape=(None, num_inputs), return_sequences=True))
        gen_model.add(GRU(units, return_sequences=False))
        gen_model.add(Dense(num_features*forecast_len, activation='tanh'))
        gen_model.add(tf.keras.layers.Reshape([forecast_len, num_features]))
        
        self.gen_model = gen_model
        
        # The discriminator object
        
        dis_model = Sequential(name=name)
        dis_model.add(Bidirectional(GRU(units,  input_shape =  (forecast_len, num_features), return_sequences=True)))
        dis_model.add(Bidirectional(GRU(units)))
        dis_model.add(Dense(1))
        
        self.dis_model = dis_model 
        
        
    def gen_loss(self, fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def dis_loss(self, fake_output, real_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    # So this is a way to create our own, custom training steps 

    @tf.function
    def train_step(self, data, generator_optimizer, discriminator_optimizer):
        data_shape = tf.shape(data)
        noise = tf.math.tanh(tf.random.normal(data_shape, mean=0.5, stddev=1.0)) # scales the data to +- 1
    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_weather = self.gen_model(noise, training=True)
            
            real_output = self.dis_model(data, training=True)
            fake_output = self.dis_model(generated_weather, training=True)
            
            generator_loss = self.gen_loss(fake_output)
            discriminator_loss = self.dis_loss(fake_output, real_output)
        
        gradients_of_generator = gen_tape.gradient(generator_loss, self.gen_model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(discriminator_loss,self.dis_model.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.gen_model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.dis_model.trainable_variables))
        
        return generator_loss, discriminator_loss
    
    # Plots the loss at each epoch 
    
    def plot_loss(self, gen_losses_epochs, dis_losses_epochs, num_epochs):
        
        # Create plot file if it doesn't exist 
        if not os.path.exists(f'plots/{self.name}'):
            os.makedirs(f'plots/{self.name}')
        
        epochs = np.arange(num_epochs+1)
    
        plt.figure()
        plt.plot(epochs, gen_losses_epochs, label='generative loss', color='blue')
        plt.plot(epochs, dis_losses_epochs, label='discriminator loss', color='orange')
        plt.legend()
        plt.savefig(f'plots/{self.name}/gen_loss_{num_epochs}.png', format='png',  bbox_inches='tight')
        
    
    def train(self, dataset, epochs, generator_optimizer, discriminator_optimizer, save_epochs=1):
        
        # Initialize checkpoint
        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
        checkpoint_savedir = os.path.join(self.checkpoint_dir, self.checkpoint_pre)
        
        gen_losses = []
        dis_losses = []
        
        gen_losses_epoch = []
        dis_losses_epoch = []
        
        for epochs in range(epochs):
            start = time.time()
        
            for data_batch in dataset:
                gen_loss, dis_loss =  self.train_step(data_batch, generator_optimizer, discriminator_optimizer)
                dis_losses.append(dis_loss)
                gen_losses.append(gen_loss)
                
            if (epochs+1) % save_epochs == 0:
                checkpoint.save(file_prefix=checkpoint_savedir) 
                
            dis_loss_epoch = dis_losses[-1]
            gen_loss_epoch = gen_losses[-1]
            
            dis_losses_epoch.append(dis_loss_epoch)
            gen_losses_epoch.append(gen_loss_epoch)

            print ('Time for epoch {} is {} sec'.format(epochs + 1, time.time()-start))
            print(f'Generative Loss: {gen_loss_epoch}, Discriminative Loss: {dis_loss_epoch}')
            self.plot_loss(gen_losses_epoch, dis_losses_epoch, epochs)
            print(f'Plotting loss to plots/{self.name} ...')
        
        # Save loss at end of training for future reference
        
        complete_loss = pd.DataFrame([gen_losses_epoch, dis_losses_epoch])
        
        
        with open(f'loss/{self.name}.csv','wb') as f:
            complete_loss.to_csv(f)
        
            
def norm_data(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

if __name__=='__main__':
    my_first_rnn = rnn_gan(80, 2,4, 'spiffy', 4, 12)
    random_noise = np.random.normal(size=(1,12,4))
    
    # normalize each input feature seperately
    
    for ii in range(4):
        random_noise[:,:,ii] = norm_data(random_noise[:,:,ii])
        
    # Let's try to output the prediction of the network 
    
    pred = my_first_rnn.gen_model(random_noise)
    
    print(my_first_rnn.gen_model(random_noise))
    
    # What decision does our discriminator make? 
    
    decision = my_first_rnn.dis_model(pred)
    

    
    
    
    
        
    
        
        
    
    
    
        
    
        
        
            
            
        
            




            
        
