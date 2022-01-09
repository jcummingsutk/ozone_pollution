import pickle
import json
import logging


def predict(input_params):
    """
    uses the model in src/model to predict the outcome variable,
    with the caveat of if the pred_prob > thresh (not necessarily .5)
    then return 1.
    input_params is a list description of the (untransformed) features in the order:
    [pregnancies, glucose, bloodpressure, insulin, bmi, diabetespedigreefunction
    age]
    """
    try:
        with open("src/conf/threshold_data.json", "r", encoding="utf-8") as read_file:
            data = json.load(read_file)
            thresh = data["thresh"]
    except IOError:
        thresh = 0.5
        logging.info("using default threshold of {:.2f}".format(thresh))
    model = pickle.load(open("src/models/model.pkl", "rb"))
    ss = pickle.load(open("src/models/scaler.pkl", "rb"))
    transformed_params = ss.transform(input_params)
    probabilities = model.predict_proba(transformed_params)
    print("the probabilities are {}".format(probabilities))
    if probabilities[0][1] > thresh:
        return 1
    else:
        return 0
