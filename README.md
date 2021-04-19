![alt text](https://github.com/newH1VE/Fish4Phish/blob/main/icon.png?raw=true)

# Fish4Phish

This project contains all code produced for my master thesis: Detection of Clone-Phishing using Machine Learning. 


# Final Approach

The most important directory. It contains the code to analyze a url using the multifilter approach evaluated in the master thesis. Classification of a url is done by:

- blacklist to check if entry is still in blacklist
- lexical filter with Random Forest
- content filter with Random Forest
- signature filter to check signature by Bing
- score fusion to predict the final classification using a Decision Tree

# Menu

            +++ MENU FOR COMMANDS +++

- database      -->  print menu for database creation.
- features      -->  print menu for feature extraction.
- filter        -->  print menu for all filter.
- predict [url] -->  predict url with final multi filter approach using two Random Forests and Decision Tree for score fusion
- test          -->  run test code from /testing.
- config        -->  print configuration from definitions file.
- exit          -->  exit the system.

Typ in the displayed command to go further in menu structure or predict a url with the final multi filter approach (98%) accuracy.

[SYSTEM] | [INFO]  | [15/04/2021 15:02:25] | [Function get_f1] Precision: 0.982420554428668

[SYSTEM] | [INFO]  | [15/04/2021 15:02:25] | [Function get_f1] Recall: 0.9764784946236559

[SYSTEM] | [INFO]  | [15/04/2021 15:02:25] | [Function get_f1] F1: 0.9794405123019886

## Example:

Type in a command:
>> predict https://github.com/newH1VE/Fish4Phish/edit/main/README.md

or

>> filter

# Project Structure

# components

Components contain the workflow to be done for different taks.

1. comp_database: workflow to create databases for all filters
2. comp_feature_extraction: workflow to create lexical/content/siganture features from created database file
3. comp_feature selection: workflow to select extracted features for lexical or content based analysis using NECGT-MI

# modules

Modules are used by there coresponding components. They contain the methods to implement the workflow of the components.

1. mod_database: all methods to delete, open, write files containing data
2. mod_feature_extraction: all methods to extract features for lexical/content/signature filter (methods for lists or single entries are seperated)
3. mod_feature_selection: all methods to select features using the neighborhood-entropy based cooperative game theory

# config

This directory contains made configurations including the definitions of paths for data and main files or parameters of the implemented logger.

1. configuration: all path/file and logging parameter definitions
2. program_config: all methods to save configs to fish4phish.ini
3. fish4phish.ini: configuration file that contains the date of the last update for the blacklist database

# definitions

This directory implements all needed classes and enums. Classes contain variables to save all features as well as the url and label for all filters. Enums specify logging actions like informative, warning and error.

## classes

all classes for blacklist, content filter, lexical filter, signature filter, letter frequencies, logging color and to save done redirects of website.

## enums

They define different actions of the logger. Three actions are implemented:
1. Informative: [INFO]
2. Warning: [WARN]
3. Error: [ERR]

# Main Files

outsourced code of main.py to make the main file slightly smaller. The main files call the workflow of the components.

- main_config: main file for menu item **config**
- main_databse: main file for menu item **database** (comp_database)
- main_features: main file for menu items **feature** contain functions for feature extraction and selection (comp_feature_extraction, comp_feature_selection)

## Main Files Filters

- main_content: main file for content based filter (phishing_filter/ml_content)
- main_lexical: main file for lexical based filter (phishing_filter/ml_lexical)

# Helper

Helpers are all functions that can not be clearly assigned to one module or are remove from a module to make the code smaller.

## Feature Helper

All methods helping the feature extraction

## Helper

All methods helping other modules than feature extraction

## Logger

All methods to log function prints and typed commands.

## Menu

Contains all print statements for menus.

# Phishing Filters

This directory contains all files for the filters. 

## Single Filter

Tested single filter approach.

## Blacklist

Implements all actions to update the blacklist or add as well as remove and check entries.

## Lexical and Content Filter

**ML_Lexical** and **ML_content** have the same structure:

- files for each machine learning modell (Random Forest: rf.py, Extreme Gradient Boosting: xgb.py, K-Nearest Neighbor: knn.py, Logistic Regression: lr.py, Support Vector Machine: svm.py, Decision Tree: dt.py, Adaptive Boosting: ab.py)

The structure for each modell is identical:

1. train_model: train the modell by the passed function parameter data
2. optimize: optimize model hyper parameters using randomized search
3. print_scores: do cross validation for 5 splits and print produced scores
4. transform_data: delete columns that don't contain features or labels (ID, URL, Final URL)
5. save_last_score: save produced score to file in folder **saved_scores**
6. load_last_score: load score from file in folder **saved_scores**
7. save_model: save model to file in folder **saved_models**
8. load_model: load model from file in folder **saved_models**
9. predict_url: predict url by model

## Score Fusion

Contains the Decision Tree with structure explained above and fusion implementing majority vote and weighted majority vote.

## Website Signature

The file **signature_check** is an implementation that inherits the Classifier class by sklearn to implement own classifiers that are compatible with sklearn functions. The file contains the signature based filter.

# Main.py

The main file of the priject that starts first and calls all explained functionalitities.

