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

def format_confusion_matrix(cm, labels):
    """Plain-text table, used by the UI and by experiments.py."""
    labels = [str(l) for l in labels]
    w = max(len(l) for l in labels) + 2
    corner = "Actual \\ Pred"
    lines = [corner.ljust(len(corner) + 2) + "".join(l.rjust(w) for l in labels)]
    for i, l in enumerate(labels):
        lines.append(l.ljust(len(corner) + 2) + "".join(str(v).rjust(w) for v in cm[i]))
    return "\n".join(lines)