# Predicting atmospheric turbulence for secure quantum communications in free space

We present the code used for the training and subsequent analysis of the recurrent neural network (RNN) model TAROQQO, used to forecast turbulence in free-space links. The results of our model, which considers free-space channels over the city of Ottawa, are featured in our preprint on arXiv (https://arxiv.org/abs/2406.14768). 

## How to use
- nn_architecture.py implements the RNN architectures to be trained.
- train.py prepares the dataset and trains the network.
- ForecastWindow.ipynb analyzes the model prediction on consecutive datasets.
- FeatureImportance.ipynb calculates the importance of features using the Permutation Feature Importance (PFI) technique. 

Use of the train.py script requires configuration of the 'train.yaml' configuration file; it can be configured to train networks of arbitrary complexity, as well as to adjust the input series duration & output time resolution. 

![alt text][logo]

[logo]: https://github.com/TareqJ1000/TurbulentNetwork/blob/tareq/TaroccoAI.png "TAROCCO Card"


