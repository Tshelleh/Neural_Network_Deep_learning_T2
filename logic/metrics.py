import numpy as np


def confusion_matrix(y, y_pred ,n_classes=3):

    if y.shape != y_pred.shape:
        raise ValueError("y and y_pred must have the same shape")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for actual,pred in zip(y,y_pred):
        cm[actual][pred] +=1
    return cm

def accuracy(cm):
    return np.trace(cm) / np.sum(cm)