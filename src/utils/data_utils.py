import pandas as pd

# import numpy as np
import logging
from pickle import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.parameter_loader import load_test_fraction


def clean_data(df) -> pd.DataFrame:
    """
    Lowercases the column headers, drops the skin thickness column and
    imputes missing data in the glucose, bloodpressure, insulin, and bmi column
    """



def scale_data(df) -> pd.DataFrame:
    """
    Scales the data using a standard scaler. Also stores it for later predictions.
    """



def load_data() -> pd.DataFrame:
    """
    Loads, cleans, and scales the diabetes data
    """



def pop_train_test_split(df,in_random_state=0):
    """
    Pops the outcome variable and train,test,splits
    """
