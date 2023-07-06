import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

def _balanced_log_loss(y_true: pd.Series, y_pred: pd.Series, **kwargs):
    # Extracting class labels from y_true
    y_true = y_true.astype(int)
    if len(y_pred.shape) == 1:
        y_pred = np.array((1-y_pred, y_pred)).T
    
    # Computing the number of observations for each class
    N0 = np.sum(y_true == 0)
    N1 = np.sum(y_true == 1)
    
    # Calculating the inverse prevalence weights
    w0 = 1 / N0
    w1 = 1 / N1
    
    # Rescaling the predicted probabilities
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    
    # Calculating the logarithmic loss for each class
    log_loss_0 = np.sum((1-y_true) * np.log(y_pred[:, 0])) / N0
    log_loss_1 = np.sum(y_true * np.log(y_pred[:, 1])) / N1
    
    # Computing the balanced logarithmic loss
    balanced_log_loss = (-w0 * log_loss_0 - w1 * log_loss_1)/(w0+w1)
    
    return balanced_log_loss

def get_bal_log_loss():
    return make_scorer(_balanced_log_loss, 
                       greater_is_better=False,
                       needs_proba=True)
