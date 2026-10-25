import os
from activations import *
import numpy as np
from logic.data_loader import load_data, split_data
from logic.preprocessing import preprocessing


class neural_network:
    FEATURES = 5
    CLASSES = 3
    WEIGHTS = None
    BIAS = None
    activations=None                      # output of each layer

    NUM_HIDDEN_LAYERS=None
    NUM_NEURONS=None                     # must be list of int
    ETA=None
    EPOCHS=None
    ACTIVATION_FUNCTION=sigmoid           # activation function: sigmoid, tanh, ReLU:
    USE_BIAS=False

    def __init__(self,num_hidden_layers, num_neurons, eta, epochs, activation_function=sigmoid, use_bias=False):
        np.random.seed(42)
        self.NUM_HIDDEN_LAYERS=num_hidden_layers
        self.NUM_NEURONS=num_neurons
        self.ETA=eta
        self.EPOCHS=epochs
        self.ACTIVATION_FUNCTION=activation_function
        self.USE_BIAS=use_bias

        self.WEIGHTS=[]
        self.BIAS=[]
        self.LAYERS=[self.FEATURES,*self.NUM_NEURONS,self.CLASSES]
        for i in range(len(self.LAYERS)-1):
            self.WEIGHTS.append(np.random.randn(self.LAYERS[i], self.LAYERS[i+1]) * 0.5)
            if self.USE_BIAS:
               self.BIAS.append(np.random.rand(1, self.LAYERS[i+1])*0.5)
            else:
               self.BIAS.append(np.zeros((1, self.LAYERS[i+1])))


    def forward(self, x):
        self.activations=[x]    # index 0 for original input

        # hidden layers
        YH = []  # output of each hidden layer
        yh = x
        for i in range(self.NUM_HIDDEN_LAYERS):
            net = np.dot(yh, self.WEIGHTS[i]) + self.BIAS[i]
            self.activations.append(net)
            yh = self.ACTIVATION_FUNCTION(net)
            YH.append(yh)
        # output layer
        net = np.dot(YH[-1], self.WEIGHTS[-1]) + self.BIAS[-1]
        self.activations.append(net)
        yk = self.ACTIVATION_FUNCTION(net)
        return yk ,YH

    def backward(self, y, yk):
        DASH_ACTIVATION_FUNCTION = globals()[self.ACTIVATION_FUNCTION.__name__ + "_dash"]

        #output layer
        sk = (y - yk) * DASH_ACTIVATION_FUNCTION(self.activations[-1])

        #hidden layers
        SH=[]
        sh=sk
        for i in range(self.NUM_HIDDEN_LAYERS-1,-1,-1):
            sh = np.dot(sh, self.WEIGHTS[i+1].T) * DASH_ACTIVATION_FUNCTION(self.activations[i+1])
            SH.append(sh)
        SH.reverse()

        return sk,SH

    def update_weights(self, x, YH,sk,SH):
            inputs = [x] + YH
            deltas = SH + [sk]
            for i in range(len(self.WEIGHTS)):
                self.WEIGHTS[i] += self.ETA * np.dot(inputs[i].T, deltas[i])
                if self.USE_BIAS:
                    self.BIAS[i] += self.ETA * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, x, y):
        # Convert to numpy arrays if needed
        x = np.asarray(x.values if hasattr(x, 'values') else x, dtype=np.float64)
        y = np.asarray(y.values if hasattr(y, 'values') else y, dtype=np.float64)

        for i in range(self.EPOCHS):
           yk, YH = self.forward(x)
           sk, SH = self.backward(y, yk)
           self.update_weights(x, YH, sk, SH)

    def predict(self, x):
        x = np.asarray(x.values if hasattr(x, 'values') else x, dtype=np.float64)
        yk , _ = self.forward(x)
        # set maximum class from each row as 1 and rest as 0
        y_predict=np.zeros_like(yk)
        y_predict[np.arange(len(yk)), np.argmax(yk, axis=1)] = 1
        return y_predict

if __name__ == "__main__":
    Base_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(Base_DIR, '..', 'data/penguins.csv')
    df = load_data(csv_path)
    train, test = split_data(df)
    X_train, y_train, X_test, y_test, Target_encoder, OriginLocation_encode = preprocessing(train, test)
    Back_Pro = neural_network(2,[3,4],0.01,100,sigmoid,True)
    Back_Pro.train(X_train, y_train)
    y_pred = Back_Pro.predict(X_test)
    acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test.values, axis=1))
    print("Accuracy:", acc)
    print(y_test,y_pred)
