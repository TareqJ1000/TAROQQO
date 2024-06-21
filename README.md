# Predicting atmospheric turbulence for secure quantum communications in free space

We present the code used for the training and subsequent analysis of the recurrent neural network (RNN) model TAROCCO, which can be used to forecast turbulence in free-space links. The results of our model, which considers free-space channels over the city of Ottawa, is featured in our preprent (arxiv link). 

# How to use
- nn_architecture.py implements the RNN architectures to be trained.
- train.py prepares the dataset and trains the network
- ForecastWindow.ipynb analyzes the model prediction on consective datasets.

Use of the train.py script requires configuration of the 'train.yaml' configuration file; it can be configured to train networks of arbitrary complexity, as well as to adjust the input series duration & output time resolution. 







