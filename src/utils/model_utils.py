import numpy as np
from src.utils.parameter_loader import (
    load_gridsearch_parameters,
    load_gridsearch_model_parameters,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json
import pickle


def find_optimal_model(X_train, y_train):
    """
    Uses a grid search to find the best gradient boosting classifier (optimizing f1 score)
    then finds the highest threshold thresh for which when the model predicts
    1 when P(input > thresh) and 0 otherwise, we achieve 80% recall
    on the training set.
    """
