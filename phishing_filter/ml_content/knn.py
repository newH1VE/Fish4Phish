# STANDARD LIBARIES
import pickle


# THIRD PARTY LIBARIES
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate


# LOCAL LIBARIES
from config.program_config import DATA_PATH, CONTENT_FEATURE_DATABASE, SCORE_PATH_CONTENT, SAVED_MODELS_PATH_CONTENT, INFO
from helper.logger import log, log_module_start, log_module_complete
from components.modules.mod_feature_extraction import extract_features_from_website


MODEL_NAME = "K-NEAREST NEIGHBOR"
SCORE_FILE = SCORE_PATH_CONTENT + "knn_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_CONTENT + "knn_model.sav"


# train the model
def train_model(do_optimize=False, data=pd.DataFrame()):

    log_module_start(MODULE_NAME=MODEL_NAME)
    pd.set_option('display.max_columns', None)

    if len(data) == 0:
        data = pd.DataFrame(pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE))
        data = transform_data(data)


    data = data.replace("True", 1)
    data = data.replace("False", 0)

    y = data['Label'].copy()
    x = data.drop(["Label"], axis=1).copy()
    train, test = train_test_split(data, test_size=0.35)
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Data ready for use.")

    y_train = train['Label'].copy()
    x_train = train.drop(["Label"], axis=1).copy()
    
    params = {
        'n_neighbor': 9
    }

    
    knn = KNeighborsClassifier()#params)
    f1 = print_scores(knn, x, y)
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Starting training.")
    
    
    knn.fit(x_train, y_train)
    save_model(knn=knn)
    log_module_complete(MODULE_NAME=MODEL_NAME)

    return f1


# transform the data
def transform_data(data):
    # drop unrelevant columns
    drop_elements = ['ID', 'URL', 'Final URL']  # drop because of heatmap for last 2
    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed


# function to print score after training
def print_scores(knn, x, y, params=None):
    if params is None:
        scores = cross_validate(knn, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True)
    else:
        scores = cross_validate(knn, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True, fit_params=params)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] New saved_scores achieved.")

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] F1-Score Test: {m}".format(m=f1_test))

    load_last_score()
    save_last_score(prec_train, prec_test, recall_train, recall_test, f1_train, f1_test)

    return f1_test


# save achived score
def save_last_score(prec_train, prec_test, recall_train, recall_test, f1_train, f1_test):
    with open(SCORE_FILE, 'w') as file:
        file.write("Precision Train: " + str(prec_train) + "\n")
        file.write("Precision Test: " + str(prec_test) + "\n")
        file.write("Recall Train: " + str(recall_train) + "\n")
        file.write("Recall Test: " + str(recall_test) + "\n")
        file.write("F1-Score Train: " + str(f1_train) + "\n")
        file.write("F1-Score Test: " + str(f1_test) + "\n")
        file.close()

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: score saved in [{f}]".format(f=SCORE_FILE))
    return


# load last score for model
def load_last_score():
    with open(SCORE_FILE, 'r') as file:
        data = file.read()

    data = data.splitlines()
    prec_train = data[0].split(" ")[2]
    prec_test = data[1].split(" ")[2]
    recall_train = data[2].split(" ")[2]
    recall_test = data[3].split(" ")[2]
    f1_train = data[4].split(" ")[2]
    f1_test = data[5].split(" ")[2]

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: Last score:")
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR] F1-Score Test: {m}".format(m=f1_test))


    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]
    return data

# save model
def save_model(knn):

    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(knn, file)

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: Model saved.")


# load saved model
def load_model():

    # load model from file
    try:
        knn = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    except Exception:
        return None

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: Model loaded.")

    return knn


# predict value with model
def predict_url(url):

    knn = load_model()
    x_pred = pd.DataFrame(extract_features_from_website(url, "PREDICT", True))
    x_pred = transform_data(x_pred)

    y_pred = knn.predict(x_pred)

    result = "NO RESULT"

    if str(y_pred[0]) == "0":
        result = "Benign"

    if str(y_pred[0]) == "1":
        result = "Phish"

    log(action_logging_enum=INFO, logging_text="[K-NEAREST NEIGHBOR]: Result for URL [{u}] = {l}".format(u=url, l=result))