
# STANDARD LIBARIES
import os

# LOCAL LIBARIES
from definitions.enum_def.logging import logging_enum

"""
This file contains all relevant definitions for paths, files and parameters for logging
"""

# read database files and combine into one database on system start
DATABASE_ON_START = False

#define root directory
ROOT_DIR = os.path.dirname(os.path.abspath("main.py"))

#define data path by root dir of the project
DATA_PATH = ROOT_DIR + "/data/"
DATA_BACKUP_PATH = DATA_PATH + "data_backup/"

#define location of main.py file
MAIN_FILE = os.path.join(ROOT_DIR, 'main.py')

# location for TLDs by IANA
TLD_LOC = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"

# location for backup of tld list
TLD_LOC_BACKUP = DATA_BACKUP_PATH + "tld_list_backup.txt"

# define path for machine learning parameters config
CONFIG_FILE = ROOT_DIR + "/config/fish4phish.ini"

# define directory for log files
LOGGING_PATH = ROOT_DIR + "/log/"

# define directory for for blacklist files
BLACKLIST_FILE = DATA_PATH + "blacklist.db"

# score path
SCORE_PATH_LEXICAL = ROOT_DIR + "/phishing_filter/ml_lexical/saved_scores/"
SCORE_PATH_CONTENT = ROOT_DIR + "/phishing_filter/ml_lexical/saved_scores/"
SCORE_PATH_FUSION = ROOT_DIR + "/phishing_filter/score_fusion/saved_scores/"
SCORE_PATH_SINGLE = ROOT_DIR + "/phishing_filter/single_filter/saved_scores/"

# saved ml_lexical path
SAVED_MODELS_PATH_LEXICAL = ROOT_DIR + "/phishing_filter/ml_lexical/saved_models/"
SAVED_MODELS_PATH_CONTENT = ROOT_DIR + "/phishing_filter/ml_content/saved_models/"
SAVED_MODELS_PATH_FUSION = ROOT_DIR + "/phishing_filter/score_fusion/saved_models/"
SAVED_MODELS_PATH_SINGLE = ROOT_DIR + "/phishing_filter/single_filter/saved_models/"

# files that contain data
ALEXA_FILE = "alexa.xml"
PHISHTANK_FILE = "phishtank.xml"
DATABASE = "data.xml"  # file that consists of all three lists
LEXICAL_FEATURE_DATABASE = "lexical_feature_data.csv"  # file that contains features extracted from DATABASE file
CONTENT_FEATURE_DATABASE = "content_feature_data.csv"  # file that contains features extracted from DATABASE file
SIGNATURE_FEATURE_DATABASE = "signature_feature_data.csv"  # file that contains features extracted from DATABASE file

# files with tokens for website processing
PHISHY_WORDS_FILE = "phishywords.xml"
BRAND_FILE="brands.xml"
LOGIN_WORDS_FILE="loginwords.xml"

# search engine keys
BING_SEARCH_KEY="______"

# define time format for log actions
LOGGING_TIME_FORMAT = "[%d/%m/%Y %H:%M:%S]"
LOGGING_FILE_NAME_FORMAT = "%Y-%m-%d"

# log definitions
INFO = logging_enum.INFO
WARNING = logging_enum.WARNING
ERROR = logging_enum.ERROR

# amount of days after that log files are deleted | 1 -> only todays log will stay | 0/None -> no logs stay
LOGGING_PERIOD = 1

# column name of featureentry class for the final database
LEXICAL_FEATURE_LIST_COLUMN_NAMES = ["ID", "Has IP", "Length URL", "Has Redirect", "Has At Symbol", "Has Token Netloc", "Has Subdomains", "Number Subdomains", "Has HTTPS", "Has Other Port"
    , "Has HTTPS Token", "Number Redirects", "Ratio Cap/NonCap", "Number Dots", "Length Netloc", "Number Dash Netloc"
    ,"Number Tokens Netloc", "Number Digits Netloc", "Number Digits Path", "Number PhishyTokens Netloc", "Number PhishyTokens Path", "Has Brand Subdomain"
    , "Has Brand Path", "Has Query", "Number Query Parameters", "Number Dots Netloc", "Number Underscore Netloc", "Has Valide TLD", "Number Slash Path"
    , "Number Comma", "Number Stars", "Number Semicolon", "Number Plus", "Has Javascript", "Number Equals", "Number Dash", "Has Fragment", "Number Fragment Values", "Number Ampersand"
    , "Has HTML Code", "Number Tilde", "Number Symbols", "Entropy", "Ratio Vowel/Consonant", "Has Digits Netloc", "Ratio Digit/Letter", "Number Dash Path"
    , "Cert Restlive", "Cert Created Shortly","Ratio Netloc/URL", "Ratio Path/URL", "Ratio Query/URL", "Ratio Fragment/URL",
                                     "KL Divergence", "Has Shortening", "Label", "URL", "Final URL"]


CONTENT_FEATURE_LIST_COLUMN_NAMES = ["ID", "Has Redirect", "Has Favicon", "Has Extern Content", "Number Extern Links", "Has Custom StatusBar", "Has Disabled RightClick", "Has PopUp",
                                     "Has iFrame", "Has Action", "Has Extern Action", "Has Form with POST", "Number PhishyTokens", "Has Input", "Ratio Description Sim", "Has Bond Status", "Has Freq Domain Extern",
                                     "Ratio Similarity", "Has Copyright", "Ratio Copyright Sim", "Ratio Title Sim", "Ratio Unique Links", "Number Inputs", "Has Input for Login",
                                     "Has Button", "Has Meta", "Has Hidden Element", "Number Option", "Number Select", "Number TH", "Number TR", "Number Table", "Number HREF", "Number LI", "Number UL",
                                     "Number OL", "Number DIV", "Number Span", "Number Article", "Number Paragr", "Number Checkbox", "Number Button", "Number Image", "Label", "URL", "Final URL"]


SIGNATURE_FEATURE_LIST_COLUMN_NAMES = ["ID", "URL", "Final URL", "Label", "Certificate Subject", "Entity1", "Entity2", "Entity3", "Entity4", "Entity5",
                                       "Term1", "Term2", "Term3", "Term4", "Term5"]

## Merken:# <html class data-device-type="dedicated" lang="de-DE">
# Phishing Website Dataset:
# https://archive.ics.uci.edu/ml/datasets/phishing+websites
