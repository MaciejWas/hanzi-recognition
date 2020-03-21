import torch
from sklearn.metrics import accuracy_score
import numpy as np


def average_misclassified_radicals(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    return np.sum(y_pred != y_true, axis=1).mean()

if __name__ == '__main__':
    BATCH_SIZE = 400

    y_true = np.c_[
            np.ones((BATCH_SIZE, 5)),
            np.zeros((BATCH_SIZE, 394 - 5))            
            ]
    y_pred = np.random.normal(0,2,(BATCH_SIZE, 394))
    y_pred = y_pred > 0.5 
    
    print(
            np.sum(y_pred != y_true, axis=1).mean()
            )
