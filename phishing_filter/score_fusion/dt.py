# STANDARD LIBARIES
import pickle


# THIRD PARTY LIBARIES
import pandas as pd
from sklearn import tree
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split


# LOCAL LIBARIES
from helper.feature_helper import remove_chars_from_URL
from config.program_config import DATA_PATH, INFO, SCORE_PATH_FUSION, SAVED_MODELS_PATH_FUSION
from helper.logger import log, log_module_complete, log_module_start
from components.modules.mod_feature_extraction import extract_features_from_website
from helper.helper import score_func


MODEL_NAME ="DECISION TREE"
SCORE_FILE = SCORE_PATH_FUSION + "dt_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_FUSION+ "dt_model.sav"

# decision tree tutorial
# https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset

# function to train the model
def train_model(data=pd.DataFrame()):

    # log training starting
    log_module_start(MODULE_NAME=MODEL_NAME)
    # read data and split into test and train

    if len(data) == 0:
        return None

    #train, test = train_test_split(data, test_size=0.2)
    pd.set_option('display.expand_frame_repr', False)
    # display all columns with head()
    pd.set_option('display.max_columns', None)
    y = data['Label']
    x = data.drop(['Label'], axis=1).values
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Data ready for use.")
    
    # divide data to inputs (x) and labels (y)
    y_train = data['Label']
    x_train = data.drop(['Label'], axis=1).values


    params = {
        'min_samples_split': 3, 
        'min_samples_leaf': 1,
        'random_state': 42, 
        'class_weight': 'balanced'
    }
    
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Starting training.")
    # create classifier with specifications
    decision_tree = tree.DecisionTreeClassifier()#params)
    f1 = print_scores(decision_tree, x, y)
    decision_tree.fit(x_train, y_train)
    save_model(decision_tree=decision_tree)
    # log train complete
    log_module_complete(MODULE_NAME=MODEL_NAME)

    return f1


# function to print score after training
def print_scores(decision_tree, x, y, params=None):
    if params is None:
        scores = cross_validate(decision_tree, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True)
    else:
        scores = cross_validate(decision_tree, x, y, cv=10, scoring=('f1', 'precision', 'recall'),
                                return_train_score=True, fit_params=params)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[DECISION TREE] New saved_scores achieved.")
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] F1-Score Test: {m}".format(m=f1_test))
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

    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: score saved in [{f}]".format(f=SCORE_FILE))
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

    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Last score:")
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE] F1-Score Test: {m}".format(m=f1_test))

    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]
    return data


# save model
def save_model(decision_tree):
    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(decision_tree, file)

    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Model saved.")


# load saved model
def load_model():
    # load model from file
    decision_tree = pickle.load(open(SAVED_MODEL_FILE, 'rb'))
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Model loaded.")

    return decision_tree


# predict value with model
def predict(scores):
    decision_tree = load_model()
    if isinstance(scores, list):
        list_1 = []
        list_2 = []
        list_3 = []
        for i in range(len(scores)):
            list_1.append(scores[i][0])
            list_2.append(scores[i][1])
            list_3.append(scores[i][2])
        
        x_pred = pd.DataFrame(
            {'Score 1': [list_1],
             'Score 2': [list_2],
             'Score 3': [list_3]
             })
    else:
        x_pred = scores
        
    y_pred = decision_tree.predict(x_pred)

    return y_pred

