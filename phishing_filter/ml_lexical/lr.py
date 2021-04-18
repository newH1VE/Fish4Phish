# STANDARD LIBARIES
import pickle


# THIRD PARTY LIBARIES
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# LOCAL LIBARIES
from components.modules.mod_feature_extraction import extract_features_from_URL
from config.program_config import LEXICAL_FEATURE_DATABASE, DATA_PATH, SCORE_PATH_LEXICAL, SAVED_MODELS_PATH_LEXICAL, \
    INFO
from helper.logger import log, log_module_complete, log_module_start

MODEL_NAME = "LOGISTIC REGRESSION"
SCORE_FILE = SCORE_PATH_LEXICAL + "lr_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_LEXICAL + "lr_model.sav"


# tutorial:
# https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python


# train the model
def train_model(do_optimize=False, data=pd.DataFrame()):
    log_module_start(MODULE_NAME=MODEL_NAME)
    pd.set_option('display.max_columns', None)

    if len(data) == 0:
        data = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
        data = transform_data(data)

    y = data['Label']
    x = data.drop(["Label"], axis=1)
    train, test = train_test_split(data, test_size=0.2)
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Data ready for use.")

    params = {
        'random_state': 1,
        'C': 0.1
    }

    logistic_regression = LogisticRegression()#params)  # random_state=1, C=0.10)
    f1= print_scores(logistic_regression, x, y)

    if do_optimize == True:
        optimize()

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Starting training.")

    y_train = train['Label']
    x_train = train.drop(["Label"], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train))
    logistic_regression.fit(x_train, y_train)
    save_model(logistic_regression=logistic_regression)

    log_module_complete(MODULE_NAME=MODEL_NAME)

    return f1


# function to print best depth setting
def optimize():
    log(action_logging_enum=INFO,
        logging_text="[LOGISTIC REGRESSION]: Starting to search for best complexity by cross validating different values.")
    # read data
    train = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
    train = transform_data(train)
    x_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].copy()
    scaler = preprocessing.StandardScaler().fit(x_train)
    y_train = np.where(y_train == 0, -1, 1)
    x_train = scaler.transform(x_train)

    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)
    model = LogisticRegression()

    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)
    best_model = clf.fit(x_train, y_train)

    # View best hyperparameters
    log(INFO, str('Best Penalty:', best_model.best_estimator_.get_params()['penalty']))
    log(INFO, str('Best C:', best_model.best_estimator_.get_params()['C']))

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Optimization completed.")


# transform the data
def transform_data(data):
    # drop unrelevant columns
    drop_elements = ['ID', 'URL', 'Final URL']  # drop because of heatmap for last 2
    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed


# function to print score after training
def print_scores(lr, x, y, params=None):
    if params is None:
        scores = cross_validate(lr, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True)
    else:
        scores = cross_validate(lr, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True, fit_params=params)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] New saved_scores achieved.")

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] F1-Score Test: {m}".format(m=f1_test))

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

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: score saved in [{f}]".format(f=SCORE_FILE))
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

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Last score:")
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION] F1-Score Test: {m}".format(m=f1_test))

    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]
    return data


# save model
def save_model(logistic_regression):
    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(logistic_regression, file)

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Model saved.")


# load saved model
def load_model():
    # load model from file
    logistic_regression = pickle.load(open(SAVED_MODEL_FILE, 'rb'))

    log(action_logging_enum=INFO, logging_text="[LOGISTIC REGRESSION]: Model loaded.")

    return logistic_regression


# predict value with model
def predict_url(url):
    logistic_regression = load_model()
    x_pred = pd.DataFrame(extract_features_from_URL(url, "PREDICT", True))
    x_pred = transform_data(x_pred)
    y_pred = logistic_regression.predict(x_pred)

    result = "NO RESULT"

    if str(y_pred[0]) == "0":
        result = "Benign"

    if str(y_pred[0]) == "1":
        result = "Phish"

    log(action_logging_enum=INFO,
        logging_text="[LOGISTIC REGRESSION]: Result for URL [{u}] = {l}".format(u=url, l=result))
    return result
