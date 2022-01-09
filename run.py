from src.utils.data_utils import load_data, pop_train_test_split
from src.utils.model_utils import find_optimal_model
import pickle
from src.models.predict import predict


def train_model():
    """
    retrain the gradient boosting classifier models
    """
    df_scaled = load_data()
    X_train, _ , y_train, _ = pop_train_test_split(df_scaled)
    grid_clf_boost = find_optimal_model(X_train, y_train)
    pickle.dump(grid_clf_boost, open("src/models/model.pkl", "wb"))

def make_prediction(input_parameters):
    return predict(input_parameters)

#transformed_example = ss.transform([[6, 148, 72, 0, 33.6, 0.627, 50]])
#make_prediction([[6, 148, 72, 0, 25, 0.627, 50]])

if __name__ == "__main__":
    train_model()
    #pred = make_prediction([[6, 148, 72, 125, 40, 0.627, 50]])
    #print("The test prediction is {}".format(pred))

#print(model.predict_proba(transformed_example))
