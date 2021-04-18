# STANDARD LIBARIES
import pickle
import sys
import linecache


# THIRD PARTY LIBARIES
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


# LOCAL LIBARIES
from components.modules import mod_feature_extraction
from config.program_config import DATA_PATH, CONTENT_FEATURE_DATABASE, SCORE_PATH_CONTENT, SAVED_MODELS_PATH_CONTENT, INFO, WARNING
from helper.logger import log_module_complete, log_module_start, log


MODEL_NAME="ADAPTIVE BOOSTING"
SCORE_FILE = SCORE_PATH_CONTENT + "ab_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_CONTENT + "ab_model.sav"


# train the model
def train_model(do_optimize=False, data=pd.DataFrame()):
    log_module_start(MODULE_NAME=MODEL_NAME)

    if len(data) == 0:
        data = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)
        data = transform_data(data)

    train, test = train_test_split(data, test_size=0.2)
    pd.set_option('display.max_columns', None)
    y = data['Label']
    x = data.drop(['Label'], axis=1).values
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Data ready for use.")

    if do_optimize == True:
        optimize()

    params = {
        'n_estimators': 120,
        'random_state': 0
    }

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Starting training.")
    adaptive_boosting = AdaBoostClassifier()#params)
    f1 = print_scores(adaptive_boosting, x, y)

    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1)

    # random_state=1, n_estimators=120, min_samples_leaf=5, min_samples_split=10,
    #                                    max_features='sqrt', max_depth=17

    adaptive_boosting.fit(x_train, y_train)
    save_model(adaptive_boosting=adaptive_boosting)
    log_module_complete(MODULE_NAME=MODEL_NAME)

    return f1


# function to print best depth setting
def optimize():
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Starting to search for best number of trees by cross validating different values.")
    # read data
    train = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)
    train = transform_data(train)
    x_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].copy()

    # Create regularization penalty space
    n_estimators = [40,50,60, 80, 100, 120, 140, 160]
    min_samples_leaf = [1,2,3,4,5]
    min_samples_split = [3,4,5,6,7,8,9,10]
    max_features = [0.1,0.15, 0.2, 0.25, 0.3, 0.35]

    # Create hyperparameter options
    hyperparameters = dict(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, max_features=max_features)
    model = AdaBoostClassifier()
    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(model, hyperparameters, cv=10, verbose=0)
    best_model = clf.fit(x_train, y_train)

    #View best hyperparameters
    #Best estimators: 60
    #Best samples leaf: 1
    #Best samples split: 3
    # Best features: 5
    log(INFO, str('Best estimators:', best_model.best_estimator_.get_params()['n_estimators']))
    log(INFO, str('Best samples leaf:', best_model.best_estimator_.get_params()['min_samples_leaf']))
    log(INFO, str('Best samples split:', best_model.best_estimator_.get_params()['min_samples_split']))
    log(INFO, str('Best features:', best_model.best_estimator_.get_params()['max_features']))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Optimization completed.")


# transform the data
def transform_data(data):
    # drop unrelevant columns
    drop_elements = ['ID', 'URL', 'Final URL']  # drop because of heatmap for last 2

    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed



# function to print score after training
def print_scores(random_forest, x, y, params=None):
    if params is None:
        scores = cross_validate(random_forest, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True)
    else:
        scores = cross_validate(random_forest, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True, fit_params=params)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] New saved_scores achieved.")

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] F1-Score Test: {m}".format(m=f1_test))

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

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: score saved in [{f}]".format(f=SCORE_FILE))
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

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Last score:")
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING] F1-Score Test: {m}".format(m=f1_test))


    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]
    return data

# save model
def save_model(adaptive_boosting):

    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(adaptive_boosting, file)

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Model saved.")


# load saved model
def load_model():

    # load model from file
    try:
        adaptive_boosting = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    except Exception:
        return None
    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Model loaded.")

    return adaptive_boosting


adaptive_boosting_pre_loaded = load_model()


# predict value with model
def predict_url(url):
    try:
        features = mod_feature_extraction.extract_features_from_website(url, "PREDICT", True)
        x_pred = pd.DataFrame(features)
        x_pred = x_pred.drop(["Label", "ID", "URL"], axis=1)
        y_pred = adaptive_boosting_pre_loaded.predict(x_pred)
        result = "NO RESULT"
        int_result = 0

        if str(y_pred[0]) == "0":
            result = "Benign"

        if str(y_pred[0]) == "1":
            result = "Phish"

    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        log(INFO, 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        log(action_logging_enum=WARNING, logging_text=str(e))
        log(action_logging_enum=WARNING, logging_text=str(e.__traceback__))
        return 0

    log(action_logging_enum=INFO, logging_text="[ADAPTIVE BOOSTING]: Result for URL [{u}] = {l}".format(u=str(url), l=result))
    return result