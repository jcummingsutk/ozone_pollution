from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

N_DEG = 5
TEST_FRACTION = 0.25
RAND_STATE = 0
CV = 7
N_JOBS = 6


def train_model():
    """
    train the model
    """
    df_weather = load_weather_data()
    df_ozone = load_ozone_data()
    df_col = df_ozone.merge(
        df_weather,
        left_on="datetime",
        right_on="datetime_dt",
        suffixes=("_oz", "_weather"),
        how="left",
    )
    df_col = engineer_features(df_col)

    use_cols_numerical = [
        "temp",
        "humidity",
        "windspeed",
        "month_x",
        "month_y",
        "sealevelpressure",
        "hour_x",
        "hour_y",
    ]
    use_cols_cat = ["type of day"]  # type of day indicates if it is a weekend or not
    target = "sample measurement"
    df_col_useful = df_col[use_cols_numerical + use_cols_cat + [target]]
    df_col_useful = df_col_useful.dropna()

    df_scaled_p, df_cat_p, y = get_poly_scaled(
        df_col_useful, use_cols_numerical, use_cols_cat, target, N_DEG
    )
    df_cat_encoded = pd.get_dummies(df_cat_p, drop_first=True)
    X = pd.concat([df_scaled_p, df_cat_encoded], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RAND_STATE
    )



    alpha = np.linspace(.05, .5, 50)
    ridge_params = {'alpha':alpha}
    reg_ridge = linear_model.Ridge()
    grid_ridge = GridSearchCV(reg_ridge, param_grid=ridge_params, scoring='neg_mean_absolute_error', cv=CV, n_jobs=N_JOBS, verbose=2)
    grid_ridge.fit(X_train, y_train)

    print(grid_ridge.score(X_train, y_train))
    print_errors(grid_ridge, X_test, y_test)
    pickle.dump(grid_ridge, open("model.pkl", "wb"))


def predict(input_dict):
    """ returns the predicted ozone pollution ppm based on input variables
    The input dictionary should be of the following form:
    {
        "temp": temperature in F
        "humidity": relative humidity as a percentage
        "windspeed": wind speed in mph
        "month": the month number, 1= Jan, 12=Dec
        "sea_level_pressure": pressure in millibars
        "hour": hour of the day, 0=midnight, 23 = 11 p.m.
    } """
    poly_pickle = pickle.load(open("poly.pkl", "rb"))
    scaler_pickle = pickle.load(open("scaler.pkl", "rb"))
    model_pickle = pickle.load(open("model.pkl", "rb"))

    hour_x, hour_y = cyclicize(input_dict["hour"], 24)
    month_x, month_y = cyclicize(input_dict["month"], 12)

    input_dict_num = {
        "temp": [input_dict["temp"]],
        "humidity": [input_dict["humidity"]],
        "windspeed": [input_dict["windspeed"]],
        "month_x": [month_x],
        "month_y": [month_y],
        "sea_level_pressure": [input_dict["sea_level_pressure"]],
        "hour_x": [hour_x],
        "hour_y": [hour_y]
    }

    input_dict_cat = {"is_weekend": [input_dict["is_weekend"]]}
    #print(input_dict_num.keys())
    input_series_num = pd.DataFrame.from_dict(input_dict_num)
    input_series_cat = pd.DataFrame.from_dict(input_dict_cat)



    poly_series = poly_pickle.transform(input_series_num)
    scaled_poly_series = scaler_pickle.transform(poly_series)
    df_poly = pd.DataFrame(scaled_poly_series)
    df_poly.columns = poly_pickle.get_feature_names(input_series_num.columns)


    input_series = pd.concat([df_poly, input_series_cat], axis=1)

    prediction = model_pickle.predict(input_series)
    print(prediction)


# transformed_example = ss.transform([[6, 148, 72, 0, 33.6, 0.627, 50]])
# make_prediction([[6, 148, 72, 0, 25, 0.627, 50]])

if __name__ == "__main__":
    #train_model()
    input_dictionary = {
        "temp": 110,
        "humidity": 0,
        "windspeed": 0,
        "month": 7,
        "sea_level_pressure": 1020.8,
        "hour": 18,
        "is_weekend": 1
    }
    pred = predict(input_dictionary)

    # print("The test prediction is {}".format(pred))

# print(model.predict_proba(transformed_example))
