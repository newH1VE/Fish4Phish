
# LOCAL LIBARIES
from config import program_config as df


# print main menu
def print_menu():
    print("\n[+++ MENU FOR COMMANDS +++]\n")
    print("database     -->  print menu for database creation.")
    print("features     -->  print menu for feature extraction.")
    print("filter       -->  print menu for all filter.")
    print("test         -->  run test code from /testing.")
    print("config       -->  print configuration from definitions file.")
    print("exit         -->  exit the system.")


def print_filter():
    print("\n[+++ MENU FOR FILTER +++]\n")
    print("back           -->  back to main menu.")
    print("lexical        -->  print menu for machine learning for lexical filter.")
    print("content        -->  print menu for machine learning for content filter.")
    print("identity       -->  print menu for website identity identification.")
    print("score fusion   -->  print menu for score fusion.")


# print database menu
def print_database(do_download_alexa, do_download_phish, do_query_alexa, check_status_phishing, check_status_benign):
    print("\n[+++ MENU FOR DATABASE CREATION +++]\n")
    print("back                         -->  back to main menu.")
    print("set query [true/false]       -->  set query for login page of entries in alexa list. (Current: {})".format(do_query_alexa))
    print("check status [phish/benign]  -->  checks status of websites by reqesting. (CURRENT: Phishing: {}, Benign: {})".format(str(check_status_phishing), str(check_status_benign)))
    print("set download [list]          -->  set download for lists from the internet. (Alexa: {}, Phishtank {})".format(do_download_alexa, do_download_phish))
    print("run                          -->  run database creation.")


# print feature menu
def print_feature_extract(signature, lexical, content, val_sets):
    print("\n[+++ MENU FOR FEATURE EXTRACTION +++]\n")
    print("back                         -->  back to main menu.")
    print("set content   [true/false]   -->  do feature extraction for content filter. (CURRENT: {})".format(content))
    print("set lexical   [true/false]   -->  do feature extraction for lexical filter. (CURRENT: {})".format(lexical))
    print("set signature [true/false]   -->  do feature extraction for signature filter. (CURRENT: {})".format(signature))
    print("generate val-sets            -->  generate test and validation set of feature data. (CURRENT: {})".format(val_sets))
    print("run                          -->  run feature extraction.")


# print feature menu
def print_features():
    print("\n[+++ MENU FOR FEATURES +++]\n")
    print("back       -->  back to main menu.")
    print("extract    -->  go to feature extraction menu.")
    print("select     -->  go to feature selection menu.")



# print feature menu
def print_feature_select(lexical, content):
    print("\n[+++ MENU FOR FEATURE SELECTION +++]\n")
    print("back                         -->  back to main menu.")
    print("select content [true/false]  -->  do feature selection for content filter. (CURRENT: {})".format(content))
    print("select lexical [true/false]  -->  do feature selection for lexical filter. (CURRENT: {})".format(lexical))
    print("run                          -->  run feature extraction.")


# print model menu for lexical filter
def print_lexical_models():
    print("\n[+++ MENU FOR MODELS OF LEXICAL FILTER +++]\n")
    print("back    -->  back to main menu.")
    print("dt      -->  print decision tree menu.")
    print("rf      -->  print random forest menu.")
    print("svm     -->  print support vector machine menu.")
    print("lr      -->  print logistic regression menu.")
    print("knn     -->  print k-nearest neighbor menu.")
    print("xgb     -->  print extreme gradient boosting menu.")
    print("ab      -->  print adaptive boosting menu.")


# print model menu
def print_content_models():
    print("\n[+++ MENU FOR MODELS OF CONTENT FILTER +++]\n")
    print("back     -->  back to main menu.")
    print("dt       -->  print decision tree menu.")
    print("rf       -->  print random forest menu.")
    print("svm      -->  print support vector machine menu.")
    print("lr       -->  print logistic regression menu.")
    print("knn      -->  print k-nearest neighbor menu.")
    print("xgb      -->  print extreme gradient boosting menu.")
    print("ab       -->  print adaptive boosting menu.")


# print menu for website identity identification
def print_identity():
    print("\n[+++ MENU FOR WEBSITE IDENTITY IDENTIFICATION +++]\n")
    print("back    -->  back to main menu.")
    print("dt      -->  print decision tree menu.")
    print("rf      -->  print random forest menu.")
    print("svm     -->  print support vector machine menu.")
    print("lr      -->  print logistic regression menu.")
    print("knn     -->  print k-nearest neighbor menu.")
    print("xgb     -->  print extreme gradient boosting menu.")
    print("ab      -->  print adaptive boosting menu.")


# print menu for website identity identification
def print_score_fusion():
    print("\n[+++ MENU FOR SCORE FUSION +++]\n")
    print("back    -->  back to main menu.")
    print("dt      -->  print decision tree menu.")
    print("rf      -->  print random forest menu.")
    print("svm     -->  print support vector machine menu.")
    print("lr      -->  print logistic regression menu.")
    print("knn     -->  print k-nearest neighbor menu.")
    print("xgb     -->  print extreme gradient boosting menu.")
    print("ab      -->  print adaptive boosting menu.")

# print config menu
def print_config():
    print("\n[+++ MENU FOR CONFIG +++]\n")
    print("back                         -->  back to main menu.")
    print("config                       -->  print configuration.")


# print decision tree menu
def print_dt(optimize):
    print("\n[+++ MENU FOR DECISION TREE +++]\n")
    print("back                     -->  back to main menu.")
    print("score                    -->  print last score.")
    print("train                    -->  retrain model by dataset.")
    print("optimize [true/false]    -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]           -->  predict specific url.")


# print random forest menu
def print_rf(optimize):
    print("\n[+++ MENU FOR RANDOM FOREST +++]\n")
    print("back                     -->  back to main menu.")
    print("score                    -->  print last score.")
    print("train                    -->  retrain model by dataset.")
    print("optimize [true/false]    -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]           -->  predict specific url.")


# print support vector machine menu
def print_svm(optimize):
    print("\n[+++ MENU FOR SUPPORT VECTOR MACHINE +++]\n")
    print("back                     -->  back to main menu.")
    print("score                    -->  print last score.")
    print("train                    -->  retrain model by dataset.")
    print("optimize [true/false]    -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]           -->  predict specific url.")


# print logistic regression menu
def print_lr(optimize):
    print("\n[+++ MENU FOR LOGISTIC REGRESSION +++]\n")
    print("back                    -->  back to main menu.")
    print("score                   -->  print last score.")
    print("train                   -->  retrain model by dataset.")
    print("optimize [true/false]   -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]          -->  predict specific url.")


# print k-nearest neighbor menu
def print_knn(optimize):
    print("\n[+++ MENU FOR K-NEAREST NEIGHBOR +++]\n")
    print("back                     -->  back to main menu.")
    print("score                    -->  print last score.")
    print("train                    -->  retrain model by dataset.")
    print("optimize [true/false]    -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]           -->  predict specific url.")


# print adaptive boosting menu
def print_ab(optimize):
    print("\n[+++ MENU FOR ADAPTIVE BOOSTING +++]\n")
    print("back                     -->  back to main menu..")
    print("score                    -->  print last score.")
    print("train                    -->  retrain model by dataset.")
    print("optimize [true/false]    -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]           -->  predict specific url.")


# print extreme gradient boosting menu
def print_xgb(optimize):
    print("\n[+++ MENU FOR EXTREME GRADIENT BOOSTING +++]\n")
    print("back                      -->  back to main menu.")
    print("score                     -->  print last score.")
    print("train                     -->  retrain model by dataset.")
    print("optimize [true/false]     -->  do optimize for best parameters. (Current: {})".format(optimize))
    print("predict  [url]            -->  predict specific url.")


# print exit menu
def print_exit():
    print("\n[+++ SYSTEM EXIT CALLED +++]\n")


# print values of config
def print_config_values():
    print("\n[+++ CURRENT CONFIGURATION +++]\n")
    print("path configurations:")
    print("ROOT DIRECTORY       -->  " + df.ROOT_DIR)
    print("DATA PATH            -->  " + df.DATA_PATH)
    print("DATA BACKUP          -->  " + df.DATA_BACKUP_PATH)
    print("MAIN FILE            -->  " + df.MAIN_FILE)
    print("LOGGING PATH         -->  " + df.LOGGING_PATH)
    print("\nnames of files containing data:")
    print("ALEXA FILE           -->  " + df.ALEXA_FILE)
    print("PHISHTANK FILE       -->  " + df.PHISHTANK_FILE)
    print("DATABSE FILE         -->  " + df.DATABASE)
    print("DATABASE FEATURE     -->  " + df.LEXICAL_FEATURE_DATABASE)
    print("\nlog configuration:")
    print("LOGGING TIME FORMAT  -->  " + df.LOGGING_TIME_FORMAT)
    print("LOGGING FILE NAME    -->  " + df.LOGGING_FILE_NAME_FORMAT + ".txt")
    print("LOGGING COMPONENTS   -->  INFO, WARNING, ERROR")
    print("LOGGING PERIOD       -->  " + str(df.LOGGING_PERIOD) + " day(s)") if df.LOGGING_PERIOD > 0 else print("LOGGING PERIOD       -->  ONLY LOGGING IN CONSOLE/ NO LOGGING FILES")
    print("\nlist of features:\n" + str(df.LEXICAL_FEATURE_LIST_COLUMN_NAMES))


