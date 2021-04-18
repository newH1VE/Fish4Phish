
# STANDARD LIBARIES
import sys

# THIRD PARTY LIBARIES

# LOCAL LIBARIES
from main_files import main_features, main_config, main_database
from main_files.main_filter import main_lexical, main_signature, main_content, main_score_fusion
from testing import testcode
from helper.logger import log_system_run, log
from config.program_config import WARNING, DATABASE_ON_START
from helper.helper import get_input_to_lower
from components import comp_database
from helper import menu


# EXIT VARIABLE DEFINITION
EXIT = False

# initial command setup
prompt ="INITIAL"

# log system initialisation
log_system_run()

# initial database creation if set in program_config.py
if DATABASE_ON_START:
    comp_database.run(do_download_alexa=False, do_download_phish=True, do_query_alexa=False)

# start loop until exit
while EXIT == False:
    if prompt.strip() != "": menu.print_menu()
    prompt = get_input_to_lower()

    # print exit
    if prompt.strip() == "exit":
        EXIT = True
        menu.print_exit()
        continue

    # print configuration menu
    if prompt.strip() == "config":
        main_config.run()
        continue


    # print feature menu
    if prompt.strip() == "features":
        main_features.run()
        continue

    # run tests
    if prompt.strip() == "test":
        testcode.run_test()
        continue

    # print model menu
    if prompt.strip() == "filter":

        back = False

        while back == False:
            menu.print_filter()
            prompt = get_input_to_lower()

            if prompt.strip() == "back":
                back = True
                continue

            if prompt.strip() == "lexical":
                main_lexical.run()
                continue

            if prompt.strip() == "content":
                main_content.run()
                continue

            if prompt.strip() == "identity":
                main_signature.run()
                continue

            if prompt.strip() == "score fusion":
                main_score_fusion.run()
                continue

            if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
        continue


    # print database menu
    if prompt.strip() == "database":
        main_database.run()
        continue

    if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")

# log shut down
log_system_run()

# terminate
sys.exit(0)
