import json


def load_gridsearch_parameters():
    with open("src/conf/gridsearch.json", "r", encoding="utf-8") as read_file:
        data = json.load(read_file)
        CV = data["CV"]
        OPT_ON = data["OPT_ON"]
        N_JOBS = data["N_JOBS"]
    return CV, OPT_ON, N_JOBS


def load_gridsearch_model_parameters():
    with open("src/conf/gridsearch_model_params.json", encoding="utf-8") as read_file:
        data = json.load(read_file)
    return data

def load_test_fraction():
    with open("src/conf/test_fraction.json", encoding="utf-8") as read_file_2:
        data = json.load(read_file_2)
        #print(data["TEST_FRACTION"], "TESTING")
        TEST_FRACTION = data["TEST_FRACTION"]
    return TEST_FRACTION
