import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x):
    return x * (1 - x)

def ReLU(x):
    return np.maximum(0, x)

def ReLU_dash(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_dash(x):
    return 1 - np.tanh(x) ** 2