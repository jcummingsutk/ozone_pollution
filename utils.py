import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import pickle


def load_ozone_data() -> pd.DataFrame:
    """
    Loads the ozone data
    """
    logging.info("Loading Ozone Data")
    file = "data/ozone_2019.csv"
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower() #lower and strip white space for columns
    df.columns = df.columns.str.strip()
    df = df.loc[(df["county name"] == "Denver") & (df["state name"] == "Colorado")]
    df = df[df["latitude"] == 39.751184]
    useful_cols = ["date local", "time local", "sample measurement"] #get the useful columns in the ozone data
    df = df[useful_cols]
    df["datetime"] = df["date local"] + " " + df["time local"] + ":00" #adjust the datetime column so it is easier to merge with other data
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_weather_data() -> pd.DataFrame:
    """
    Loads the weather data
    """
    logging.info("Loading Weather Data")
    file = "data/denver_weather_2019.csv"
    df = pd.read_csv(file)
    df["datetime_dt"] = pd.to_datetime(df["datetime"])
    return df

def cyclicize(in_series, T):
    """ Transforms a variable into a cylcic variable with period T by
    doing cos(2pi/T x), sin(2pi/T x). Will work for a series as well as a number
    This is exremely useful for time-variables, where for example the hour can be 0-23
    but hour 0 is close to 23. Essentially, we are mapping a line segment to a circle."""

    cyclic_x = np.cos(2*np.pi/T * in_series)
    cyclic_y = np.sin(2*np.pi/T * in_series)

    return cyclic_x, cyclic_y

def engineer_features(df_col) -> pd.DataFrame:
    """ Takes in a merged dataset of ozone and weather for colorado and
    cyclicizes the hour and month vairables as well as classifies each day as weekday or not"""
    df_col["day_of_week"] = df_col["datetime_dt"].dt.dayofweek
    df_col["type of day"] = "weekday"
    weekend_mask = (df_col["day_of_week"] == 5) | (df_col["day_of_week"] == 6)
    df_col.loc[weekend_mask, "type of day"] = "weekend"
    df_col["hour"] = df_col["datetime_dt"].dt.hour
    df_col["hour_x"], df_col["hour_y"] = cyclicize(df_col["hour"], 24)
    df_col["month"] = df_col["datetime_dt"].dt.month
    df_col["month_x"], df_col["month_y"] = cyclicize(df_col["month"], 12)
    return df_col


def get_poly_scaled(df, numerical_cols, cat_cols, target_col, n_deg):
    """Takes in a data set, the numerical columns you are interested in, and categorical columns and degree and returns
    the scaled numerical dataset with with polynomial features of degree n_deg, the categorical columns, and the target columns all properly indexed
    Args:
    in_df: dataframe
    numerical_cols: numerical column names, list
    cat_cols: categorical column names, list
    target: target name, string
    n_deg: integer, the degree
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler

    ss = StandardScaler()
    poly = PolynomialFeatures(n_deg)
    in_df = df.reset_index()
    # Ensure the index is set from 0 to n-1, since poly.fit_transform will return a dataframe and we'll have unmatched indices if we do not do this
    # in_df.reset_index(inplace=True)

    # Categorize the types of features and target
    df_num = in_df[numerical_cols]
    df_cat = in_df[cat_cols]
    y = in_df[target_col]
    # Transform
    use_poly = poly.fit_transform(df_num)
    df_poly = pd.DataFrame(use_poly)

    # Set column names
    df_poly.columns = poly.get_feature_names(df_num.columns)
    df_scaled_poly = pd.DataFrame(
        ss.fit_transform(df_poly), index=df_poly.index, columns=df_poly.columns
    )
    pickle.dump(ss, open("scaler.pkl", "wb"))
    pickle.dump(poly, open("poly.pkl", "wb"))
    return (df_scaled_poly, df_cat, y)


def get_mae(reg, X, y):
    """returns the mean average error of the regressor reg on X with true outputs y"""
    y_pred = reg.predict(X)
    abs_err = np.abs(y_pred - y)
    return abs_err.mean()


def get_rmse(reg, X, y):
    """returns the root mean square error of the regressor reg on X with true outputs y"""
    y_pred = reg.predict(X)
    sqr_err = np.power(y_pred - y, 2)
    mse = np.mean(sqr_err)
    return np.sqrt(mse)


def print_errors(reg, X, y):
    """prints mae and rmse"""
    print("MAE: {:.4f} \nRMSE: {:.4f}".format(get_mae(reg, X, y), get_rmse(reg, X, y)))
