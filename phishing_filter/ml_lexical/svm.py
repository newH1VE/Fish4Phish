# STANDARD LIBARIES
import pickle


# THIRD PARTY LIBARIES
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


# LOCAL LIBARIES
from components.modules.mod_feature_extraction import extract_features_from_URL
from config.program_config import DATA_PATH, LEXICAL_FEATURE_DATABASE, SCORE_PATH_LEXICAL, SAVED_MODELS_PATH_LEXICAL, \
    INFO
from helper.logger import log, log_module_start, log_module_complete

MODEL_NAME = "SUPPORT VECTOR MACHINE"
SCORE_FILE = SCORE_PATH_LEXICAL + "svm_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_LEXICAL + "svm_model.sav"


# train the model
def train_model(do_optimize=False, data=pd.DataFrame()):
    log_module_start(MODULE_NAME=MODEL_NAME)

    if len(data) == 0:
        data = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
        data = transform_data(data)

    y = data['Label']
    x = data.drop(['Label'], axis=1).values
    train, test = train_test_split(data, test_size=0.2)

    pd.set_option('display.max_columns', None)

    if do_optimize == True:
        optimize()

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Data ready for use.")

    # support vector machine
    g = 0.1
    c = 0.1

    params = {
        'kernel': 'linear',
        'random_state': 0,
        'gamma': g,
        'C': c
    }

    support_vector_machine = SVC()#params)  # params=params)
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Starting training.")
    f1 = print_scores(support_vector_machine=support_vector_machine, x=x, y=y)  # , params=params)

    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1).values
    support_vector_machine.fit(x_train, y_train)
    save_model(support_vector_machine=support_vector_machine)
    log_module_complete(MODULE_NAME=MODEL_NAME)
    return f1


def optimize():
    log(action_logging_enum=INFO,
        logging_text="[SUPPORT VECTOR MACHINE]: Starting to search for best c, gamma, kernel by cross validating different values.")

    # read data
    train = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
    train = transform_data(train)

    x_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].copy()

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Create grid search using 5-fold cross validation
    model = SVC()
    clf = GridSearchCV(model, param_grid, cv=10, verbose=0)
    best_model = clf.fit(x_train, y_train)

    # View best hyperparameters
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: Hyperparameter tuning completed.")
    log(INFO, str('Best Kernel:', best_model.best_estimator_.get_params()['kernel']))
    log(INFO, str('Best C:', best_model.best_estimator_.get_params()['C']))
    log(INFO, str('Best gamma:', best_model.best_estimator_.get_params()['gamma']))


# transform the data
def transform_data(data):
    # drop unrelevant columns
    drop_elements = ['ID', 'URL', 'Final URL']
    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed


# function to print score after training
def print_scores(support_vector_machine, x, y, params=None):
    if params is None:
        scores = cross_validate(support_vector_machine, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True)
    else:
        scores = cross_validate(support_vector_machine, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True, fit_params=params)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] New saved_scores achieved.")

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] F1-Score Test: {m}".format(m=f1_test))

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

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: score saved in [{f}]".format(f=SCORE_FILE))

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

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: Last score:")
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE] F1-Score Test: {m}".format(m=f1_test))

    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]

    return data


# save model
def save_model(support_vector_machine):
    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(support_vector_machine, file)

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: Model saved.")


# load saved model
def load_model():
    # load model from file
    support_vector_machine = pickle.load(open(SAVED_MODEL_FILE, 'rb'))

    log(action_logging_enum=INFO, logging_text="[SUPPORT VECTOR MACHINE]: Model loaded.")

    return support_vector_machine


# predict value with model
def predict_url(url):
    support_vector_machine = load_model()
    x_pred = pd.DataFrame(extract_features_from_URL(url, "PREDICT", True))
    x_pred = transform_data(x_pred)
    y_pred = support_vector_machine.predict(x_pred)
    result = "NO RESULT"

    if str(y_pred[0]) == "0":
        result = "Benign"

    if str(y_pred[0]) == "1":
        result = "Phish"

    log(action_logging_enum=INFO,
        logging_text="[SUPPORT VECTOR MACHINE]: Result for URL [{u}] = {l}".format(u=url, l=result))
    return result
