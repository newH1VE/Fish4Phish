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
from config.program_config import DATA_PATH, CONTENT_FEATURE_DATABASE, INFO, SCORE_PATH_CONTENT, SAVED_MODELS_PATH_CONTENT
from helper.logger import log, log_module_complete, log_module_start
from components.modules.mod_feature_extraction import extract_features_from_website
from helper.helper import score_func


MODEL_NAME ="DECISION TREE"
SCORE_FILE = SCORE_PATH_CONTENT + "dt_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_CONTENT+ "dt_model.sav"

# decision tree tutorial
# https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset

# function to train the model
def train_model(do_optimize=False, data=pd.DataFrame()):

    # log training starting
    log_module_start(MODULE_NAME=MODEL_NAME)
    # read data and split into test and train

    if len(data) == 0:
        print(len(data))
        data = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)
        # transform data
        data = transform_data(data)

    train, test = train_test_split(data, test_size=0.2)
    pd.set_option('display.expand_frame_repr', False)
    # display all columns with head()
    pd.set_option('display.max_columns', None)
    y = data['Label']
    x = data.drop(['Label'], axis=1).values
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Data ready for use.")
    
    # divide data to inputs (x) and labels (y)
    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1).values


    if do_optimize==True:
        optimize()

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

# transform the data
def transform_data(data):
    # drop unrelevant columns

    drop_elements = ['ID', 'URL', 'Final URL']  # drop because of heatmap for last 2

    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed


# function to print best depth setting
def optimize():
    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Starting to search for best depth by cross validating different values.")

    # read data
    train = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)
    # drop useless columns before training
    drop_elements = ['ID', 'URL']  # drop because of heatmap for last 2
    train = train.drop(drop_elements, axis=1)
    x_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].copy()

    # Create regularization penalty space
    class_weight = ['balanced']
    min_samples_leaf = [1, 2, 3]
    min_samples_split = [2, 3, 4]
    max_features = ['auto', 'sqrt', 5, 6]
    random_state = [42]
    max_depth = [14, 15, 16, 19, 20, 21, 22, 23, 25]

    # Create hyperparameter options
    hyperparameters = dict(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                           random_state=random_state, class_weight=class_weight, max_features=max_features, max_depth=max_depth)

    model = tree.DecisionTreeClassifier()
    # Create grid search using 5-fold cross validation
    clf = RandomizedSearchCV(model, hyperparameters, n_iter=100, cv=3, verbose=10, n_jobs=1, scoring='f1_weighted')
    best_model = clf.fit(x_train, y_train)

    # View best hyperparameters
    # Best estimators: 60
    # Best samples leaf: 1
    # Best samples split: 3
    # Best features: 5
    log(INFO, str('Best estimators:', best_model.best_estimator_.get_params()['n_estimators']))
    log(INFO, str('Best samples leaf:', best_model.best_estimator_.get_params()['min_samples_leaf']))
    log(INFO, str('Best samples split:', best_model.best_estimator_.get_params()['min_samples_split']))
    log(INFO, str('Best features:', best_model.best_estimator_.get_params()['max_features']))

    # maximum depth for number of columns
    max_depth = len(train.columns)
    print(max_depth)
    cv = KFold(n_splits=10)
    accuracies = list()
    errors = list()
    max_attributes = max_depth
    depth_range = range(10, max_attributes + 10)

    scorer = {'main': 'accuracy',
              'custom': make_scorer(score_func)}


    for depth in depth_range:
        fold_error = []
        fold_accuracy = []
        tree_model = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=15, min_samples_leaf=10,
                                                random_state=42, class_weight='balanced')

        cv_results = cross_validate(tree_model, X=x_train, y=y_train, cv=10, return_train_score=True)

        for res in cv_results['train_score']:
            error = 1 - res
            fold_error.append(error)
            fold_accuracy.append(res)

        avg_error = sum(fold_error) / len(fold_error)
        avg_accuracy = sum(fold_accuracy) / len(fold_accuracy)
        log(action_logging_enum=INFO, logging_text="AVG ERROR: {f}".format(f=avg_error))
        log(action_logging_enum=INFO, logging_text="AVG PREC: {f}".format(f=avg_accuracy))
        errors.append(avg_error)
        accuracies.append(avg_accuracy)

    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Optimization completed.")




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
def predict_url(url):

    decision_tree = load_model()
    x_pred = pd.DataFrame(extract_features_from_website(url, "PREDICT", True))
    x_pred = transform_data(x_pred)
    y_pred = decision_tree.predict(x_pred)

    result = "NO RESULT"

    if str(y_pred[0]) == "0":
        result = "Benign"

    if str(y_pred[0]) == "1":
        result = "Phish"

    log(action_logging_enum=INFO, logging_text="[DECISION TREE]: Result for URL [{u}] = {l}".format(u=url, l=result))
    return result