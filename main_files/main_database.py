# STANDARD LIBARIES

# THIRD PARTY LIBARIES

# LOCAL LIBARIES
from helper.logger import log
from config.program_config import INFO, WARNING
from helper.helper import get_input_to_lower
from components import comp_database
from helper import menu

"""
Code to translate commands to actions for component databse
"""

def run():
    back = False
    do_download_alexa = False
    do_download_phish = False
    do_query_alexa = False
    check_status_phishing = False
    check_status_benign = False

    while back == False:
        menu.print_database(do_download_alexa, do_download_phish, do_query_alexa, check_status_phishing, check_status_benign)
        prompt = get_input_to_lower()

        if prompt.strip() == "back":
            back = True
            continue

        if prompt.strip() == "run":
            comp_database.run(do_download_alexa, do_download_phish, do_query_alexa, check_status_phishing, check_status_benign)
            do_download_alexa = False
            do_download_phish = False
            do_query_alexa = False
            check_status_phishing = False
            check_status_benign = False
            continue

        if prompt.strip() == "set download alexa":
            do_download_alexa = not (do_download_alexa)
            log(action_logging_enum=INFO, logging_text="download Alexa set to: " + str(do_download_alexa))
            continue

        if prompt.strip() == "set download phishtank":
            do_download_phish = not (do_download_phish)
            log(action_logging_enum=INFO, logging_text="download Phishtank set to: " + str(do_download_phish))
            continue

        if prompt.strip().startswith("check status"):
            if prompt.__contains__("phish"):
                check_status_phishing = not check_status_phishing
                log(action_logging_enum=INFO, logging_text="check status of phishing pages set to: " + str(check_status_phishing))
                continue

            if prompt.__contains__("benign"):
                check_status_benign = not check_status_benign
                log(action_logging_enum=INFO, logging_text="check status of benign pages set to: " + str(check_status_benign))
                continue

        if prompt.strip().startswith("set query"):
            if prompt.split(" ")[2] == "true":
                do_query_alexa = True
                log(action_logging_enum=INFO, logging_text="query login pages set to: " + str(do_query_alexa))
                continue

            if prompt.split(" ")[2] == "false":
                do_query_alexa = False
                log(action_logging_enum=INFO, logging_text="query login pages set to: " + str(do_query_alexa))
                continue

        if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")


    return