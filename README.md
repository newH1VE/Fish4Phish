![alt text](https://github.com/newH1VE/Fish4Phish/blob/main/icon.png?raw=true)

# Fish4Phish

This project contains all code produced for my master thesis: Detection of Clone-Phishing using Machine Learning. 

The code is splitted in seeveral parts:

# Final Approach

The most important directory. It contains the code to analyze a url using the multifilter approach evaluated in the master thesis. Classification of a url is done by:

- blacklist to check if entry is still in blacklist
- lexical filter with Random Forest
- content filter with Random Forest
- signature filter to check signature by Bing
- score fusion to predict the final classification using a Decision Tree

# Menu

            +++ MENU FOR COMMANDS +++

database      -->  print menu for database creation.
features      -->  print menu for feature extraction.
filter        -->  print menu for all filter.
predict [url] -->  predict url with final multi filter approach using two Random Forests and Decision Tree for score fusion
test          -->  run test code from /testing.
config        -->  print configuration from definitions file.
exit          -->  exit the system.

Typ in the displayed command to go further in menu structure or predict a url with the final multi filter approach (98%) accuracy.

##Example:

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


