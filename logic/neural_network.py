import random
from activations import *
import numpy as np

np.random.seed(42)

class neural_network:
    FEATURES = 5
    CLASSES = 3
    WEIGHTS = None
    BIAS = np.random.rand()

    NUM_HIDDEN_LAYERS=None
    NUM_NEURONS=None
    ETA=None
    EPOCHS=None
    ACTIVATION_FUNCTION=None           # activation function: sigmoid:0, tanh:1, ReLU:2
    USE_BIAS=False

    def __init__(self,num_hidden_layers, num_neurons, eta, epochs, activation_function, bias=False):
        self.NUM_HIDDEN_LAYERS=num_hidden_layers
        self.NUM_NEURONS=num_neurons
        self.ETA=eta
        self.EPOCHS=epochs
        self.ACTIVATION_FUNCTION=activation_function
        self.USE_BIAS=bias
        self.WEIGHTS=np.random.rand(num_hidden_layers+1, num_neurons)