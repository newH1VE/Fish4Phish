
# THIRD PARTY LIBARIES
import ray

# LOCAL LIBARIES
from helper.logger import log, log_module_complete, log_module_start
from config.program_config import INFO, WARNING, ERROR, DATABASE, CONTENT_FEATURE_DATABASE, LEXICAL_FEATURE_DATABASE
from components.modules.mod_database import delete_data, write_lexical_features_CSV, write_content_features_CSV, open_dataset_XML_file, write_signature_features_CSV
from helper.feature_helper import binarize_labels
from components.modules import mod_feature_extraction as f
from components.modules.mod_database import generate_validation_sets

MODULE_NAME ="FEATURE EXTRACTION"

def run(content=False, lexical=False, signature=False, val_sets=False):

    # log module start
    log_module_start(MODULE_NAME=MODULE_NAME)

    content_feature_list = []
    lexical_feature_list = []

    if val_sets:
        generate_validation_sets()
        return

    # open data file and write to list
    data_list = open_dataset_XML_file(filename=DATABASE, iterateable="entry", label_label="label", url_label="url")

    if data_list == None:
        log(action_logging_enum=WARNING, logging_text="[MODULE FEATURE EXTRACTION]: CSV File [data.csv] was not found. returning ...")
        return

    # remove https:// http://
    #data_list = remove_chars_from_list(data_list)
    #log(action_logging_enum=INFO, logging_text="[MODULE FEATURE EXTRACTION]: https:// and http:// from all urls removed.")

    # binarize labels
    data_list = binarize_labels(data_list)
    log(action_logging_enum=INFO, logging_text="[MODULE FEATURE EXTRACTION]: Labels binarized")

    # create feature_list with FeatureEntries
    if lexical:
        lexical_feature_list = f.extract_features_from_URL_list(data=data_list)

    if content:
        process = True
        index = 6000
        append = False
        last_index = 5967

        if index == 0:
            delete_data(filename=CONTENT_FEATURE_DATABASE)

        ray.init(num_cpus=6)
        while process:

            end_index = index + 1000


            if end_index >= len(data_list):
                end_index = len(data_list) - 1
                process = False

            if index > 0:
                append = True

            copy_data = data_list[index:end_index]

            content_feature_list = f.extract_features_from_website_list_ray(data=copy_data)

            if not len(content_feature_list) > 0:
                log(ERROR,
                    "[MODULE FEATURE EXTRACTION]: Error while creating feature list for content filter. The list is empty")
                process = False
                break
            last_index += 1
            last_index = write_content_features_CSV(feature_list=content_feature_list, append=append, new_index=last_index)
            log(INFO, "[MODULE FEATURE EXTRACTION]: Feature list for content filter was writen.")

            index += 1000
            log(INFO, "[MODULE FEATURE EXTRACTION]: Feature list for content filter was writen. (Next for index: {}".format(index))

        ray.shutdown()

    if signature:
        ray.init(num_cpus=6)
        signature_feature_list = f.extract_features_from_signature_list(data=data_list)
        ray.shutdown()
        write_signature_features_CSV(feature_list=signature_feature_list)


    # feature extraction completeted
    log(action_logging_enum=INFO, logging_text="[MODULE FEATURE EXTRACTION]: Feature extraction completed.")


    # check whether the list has entries
    if len(lexical_feature_list) > 0 and lexical:
        log(INFO, "[MODULE FEATURE EXTRACTION]: Feature list for lexical filter successfully created.")

        delete_data(filename=LEXICAL_FEATURE_DATABASE)

        # write lexical_feature_list to csv file
        write_lexical_features_CSV(feature_list=lexical_feature_list)

    elif lexical:
        log(ERROR, "[MODULE FEATURE EXTRACTION]: Error while creating feature list for lexical filter. The list is empty")

    # log module completion
    log_module_complete(MODULE_NAME=MODULE_NAME)
