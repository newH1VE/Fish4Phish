![alt text](https://github.com/newH1VE/Fish4Phish/blob/main/icon.png?raw=true)

# Fish4Phish

This project contains all code produced for my master thesis: Detection of Clone-Phishing using Machine Learning. 

The code is splitted in seeveral parts:

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

This directory implements all needed classes and enums. Classes contain variables to save all features as well as the url and label for all filters.

## classes



