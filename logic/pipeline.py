"""Single entry point to the logic layer: the UI and experiments.py only talk to this module.
"""
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from logic.activations import sigmoid, tanh
from logic.data_loader import load_data, split_data
from logic.metrics import confusion_matrix, accuracy
from logic.neural_network import neural_network
from logic.preprocessing import NUMERIC_COLS, ORIGIN_LOCATION, preprocessing, preprocess_sample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, 'data', 'penguins.csv')

ACTIVATIONS = {"Sigmoid": sigmoid, "Tanh": tanh}
NUMERIC_FEATURES = list(ORIGIN_LOCATION)
ORIGINS = [name for name in ORIGIN_LOCATION if name != "Other"]


@dataclass
class TrainResult:
    model: neural_network
    train_accuracy: float
    test_accuracy: float
    train_cm: np.ndarray
    test_cm: np.ndarray


class Pipeline:
    def __init__(self, csv_path=DEFAULT_CSV):
        df = load_data(csv_path)
        train, test = split_data(df)
        (self.X_train, self.y_train, self.X_test, self.y_test,
         self.target_encoder, self.origin_encode, self.stats) = preprocessing(train, test)
        self.class_labels = self._class_labels(df)

    def _class_labels(self, df):
        """Species name for each output neuron (falls back to 'Class i')."""
        labels = [f"Class {i}" for i in range(neural_network.CLASSES)]
        try:
            names = list(df["Species"].unique())
            enc = self.target_encoder.transform(pd.DataFrame({"Species": names}))
            for name, idx in zip(names, np.argmax(enc.values, axis=1)):
                labels[int(idx)] = str(name)
        except Exception:
            pass
        return labels

    @staticmethod
    def validate_params(num_hidden_layers, num_neurons, eta, epochs):
        if num_hidden_layers < 1:
            raise ValueError("Number of hidden layers must be at least 1")
        if len(num_neurons) != num_hidden_layers or any(n < 1 for n in num_neurons):
            raise ValueError(f"Enter exactly {num_hidden_layers} positive integer(s) for the neurons")
        if eta <= 0:
            raise ValueError("Learning rate must be > 0")
        if epochs < 1:
            raise ValueError("Number of epochs must be >= 1")

    def train(self, num_hidden_layers, num_neurons, eta, epochs, activation="Sigmoid", use_bias=True):
        self.validate_params(num_hidden_layers, num_neurons, eta, epochs)
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(ACTIVATIONS)}")
        model = neural_network(num_hidden_layers, list(num_neurons), eta, epochs,
                               ACTIVATIONS[activation], use_bias)
        with np.errstate(all="ignore"):       # a too-large eta can overflow exp(); don't spam warnings
            model.train(self.X_train, self.y_train)
            train_cm, train_acc = self._evaluate(model, self.X_train, self.y_train)
            test_cm, test_acc = self._evaluate(model, self.X_test, self.y_test)
        return TrainResult(model, float(train_acc), float(test_acc), train_cm, test_cm)

    def classify(self, model, sample):
        """sample: dict with the 4 raw numeric features and 'OriginLocation' (name).
        Returns (class index, class name)."""
        x = preprocess_sample(sample, self.stats, self.origin_encode)
        with np.errstate(all="ignore"):
            one_hot = model.predict(x)
        idx = int(np.argmax(one_hot, axis=1)[0])
        return idx, self.class_labels[idx]

    @staticmethod
    def _evaluate(model, X, y):
        pred = np.argmax(model.predict(X), axis=1)
        true = np.argmax(np.asarray(y.values if hasattr(y, "values") else y), axis=1)
        cm = confusion_matrix(true, pred)
        return cm, accuracy(cm)