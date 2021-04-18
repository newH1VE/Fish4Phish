# STANDARD LIBARIES
import linecache
import pickle
import sys


# THIRD PARTY LIBARIES
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV


# LOCAL LIBARIES
from components.modules import mod_feature_extraction
from config.program_config import DATA_PATH, LEXICAL_FEATURE_DATABASE, SCORE_PATH_LEXICAL, SAVED_MODELS_PATH_LEXICAL, \
    INFO, WARNING
from helper.logger import log_module_complete, log_module_start, log

MODEL_NAME = "RANDOM FOREST"
SCORE_FILE = SCORE_PATH_LEXICAL + "rf_score.txt"
SAVED_MODEL_FILE = SAVED_MODELS_PATH_LEXICAL + "rf_model.sav"


# train the model
def train_model(do_optimize=False, data=pd.DataFrame()):
    log_module_start(MODULE_NAME=MODEL_NAME)

    if len(data) == 0:
        return None

    data = transform_data(data)


    train, test = train_test_split(data, test_size=0.2)
    pd.set_option('display.max_columns', None)
    y = data['Label']
    x = data.drop(['Label'], axis=1).values
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Data ready for use.")

    if do_optimize == True:
        optimize()

    params = {
        'n_estimators': 800,
        'max_features': 6,
        'max_depth': 21,
        'min_samples_leaf': 1,
        'min_samples_split': 4
    }

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Starting training.")
    random_forest = RandomForestClassifier(n_estimators=700, max_features='auto', min_samples_leaf=2, min_samples_split=3)
    f1 = print_scores(random_forest, x, y)

    #   'n_estimators': 1400,
    #    'max_features': 'sqrt',
    #    'max_depth': 20,
    #    'min_samples_leaf': 2,
    #    'min_samples_split': 4
    #}



    # random_state=1, n_estimators=120, min_samples_leaf=5, min_samples_split=10,
    #                                    max_features='sqrt', max_depth=17

    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1).values
    random_forest.fit(x_train, y_train)
    save_model(random_forest=random_forest)
    log_module_complete(MODULE_NAME=MODEL_NAME)

    return f1


# function to print best depth setting
def optimize():
    log(action_logging_enum=INFO,
        logging_text="[RANDOM FOREST]: Starting to search for best number of trees by cross validating different values.")
    # read data
    train = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
    train = transform_data(train)
    train = train[['Label', 'Entropy', 'Ratio Netloc/URL', 'Length URL', 'Ratio Digit/Letter', 'Ratio Path/URL', 'Has HTTPS', 'Length Netloc', 'KL Divergence', 'Ratio Vowel/Consonant', 'Number Symbols', 'Number Dots', 'Number Tokens Netloc', 'Number Digits Path', 'Ratio Cap/NonCap', 'Number Dash', 'Number Dash Netloc', 'Has Token Netloc', 'Number Slash Path', 'Ratio Query/URL', 'Number Digits Netloc', 'Number Redirects', 'Number PhishyTokens Path', 'Has Digits Netloc', 'Number Query Parameters', 'Number Dots Netloc', 'Has Query', 'Number Equals', 'Number Semicolon', 'Number Ampersand', 'Cert Created Shortly', 'Number Stars']]
    x_train = train.drop(['Label'], axis=1)
    y_train = train['Label'].copy()

    # Create regularization penalty space
    n_estimators = [800, 1000, 1200, 1400, 1600, 1800]
    min_samples_leaf = [1, 2, 3]
    min_samples_split = [2, 3, 4]
    max_features = ['auto', 'sqrt', 5, 6]
    max_depth = [14, 15, 16, 19, 20, 21]

    # Create hyperparameter options
    hyperparameters = dict(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                           n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)

    model = RandomForestClassifier()
    # Create grid search using 5-fold cross validation
    clf = RandomizedSearchCV(model, hyperparameters, n_iter=100, cv=3, verbose=10, n_jobs=1, scoring='f1_weighted')
    best_model = clf.fit(x_train, y_train)

    # View best hyperparameters
    # Best estimators: 60
    # Best samples leaf: 1
    # Best samples split: 3
    # Best features: 5
    print("Best estimators: {}".format(best_model.best_estimator_.get_params()['n_estimators']))
    print("Best samples leaf: {}".format(best_model.best_estimator_.get_params()['min_samples_leaf']))
    print("Best samples split: {}".format(best_model.best_estimator_.get_params()['min_samples_split']))
    print("Best features: {}".format(best_model.best_estimator_.get_params()['max_features']))
    print("Best depth: {}".format(best_model.best_estimator_.get_params()['max_depth']))

    log(INFO, str('Best estimators:', best_model.best_estimator_.get_params()['n_estimators']))
    log(INFO, str('Best samples leaf:', best_model.best_estimator_.get_params()['min_samples_leaf']))
    log(INFO, str('Best samples split:', best_model.best_estimator_.get_params()['min_samples_split']))
    log(INFO, str('Best features:', best_model.best_estimator_.get_params()['max_features']))

    estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]
    cv = KFold(n_splits=10)
    accuracies = list()
    errors = list()

    for est in estimators:
        fold_error = []
        fold_accuracy = []
        random_forest = RandomForestClassifier()#n_estimators=est, min_samples_split=3, max_features='sqrt')

        cv_results = cross_validate(random_forest, X=x_train, y=y_train, cv=10, return_train_score=True)

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

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: Optimitazion completed.")


# transform the data
def transform_data(data):
    # drop unrelevant columns
    drop_elements = ['ID', 'URL', 'Final URL']  # drop because of heatmap for last 2
    data_transformed = data.drop(drop_elements, axis=1)

    return data_transformed


# function to print score after training
def print_scores(random_forest, x, y):

    scores = cross_validate(random_forest, x, y, cv=5, scoring=('f1', 'precision', 'recall'),
                            return_train_score=True)

    prec_train = np.array(scores["train_precision"]).mean()
    prec_test = np.array(scores["test_precision"]).mean()
    recall_train = np.array(scores["train_recall"]).mean()
    recall_test = np.array(scores["test_recall"]).mean()
    f1_train = np.array(scores["train_f1"]).mean()
    f1_test = np.array(scores["test_f1"]).mean()

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] New saved_scores achieved.")

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: NEW SCORE.")
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] F1-Score Test: {m}".format(m=f1_test))

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

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: score saved in [{f}]".format(f=SCORE_FILE))
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

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: Last score:")
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Precision Train: {m}".format(m=prec_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Precision Test: {m}".format(m=prec_test))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Recall Train: {m}".format(m=recall_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] Recall Test: {m}".format(m=recall_test))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] F1-Score Train: {m}".format(m=f1_train))
    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST] F1-Score Test: {m}".format(m=f1_test))

    data = [prec_train, prec_test, recall_train, recall_test, f1_train, f1_test]
    return data


# save model
def save_model(random_forest):
    with open(SAVED_MODEL_FILE, 'wb') as file:
        pickle.dump(random_forest, file)

    log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: Model saved.")


# load saved model
def load_model():
    try:
        # load model from file
        random_forest = pickle.load(open(SAVED_MODEL_FILE, 'rb'))

        log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: Model loaded.")

        return random_forest
    except Exception:
        return None





# predict value with model
def predict_url(url):
    try:
        if isinstance(url, str):
            features = mod_feature_extraction.extract_features_from_URL(url, "PREDICT", True)
            x_pred = pd.DataFrame(features)
        else:
            x_pred = url

        x_pred = transform_data(x_pred)


        random_forest = load_model()
        y_pred = random_forest.predict(x_pred)

        return y_pred.tolist()

    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        log(action_logging_enum=WARNING, logging_text=str(e))
        log(action_logging_enum=WARNING, logging_text=str(e.__traceback__))
        return None

    # log(action_logging_enum=INFO, logging_text="[RANDOM FOREST]: Result for URL [{u}] = {l}".format(u=str(url), l=result))

