# STANDARD LIBARIES

# THIRD PARTY LIBARIES

# LOCAL LIBARIES
from helper.logger import log
from config.program_config import WARNING
from helper.helper import get_input_to_lower
from helper import menu


# main code for config menu
def run():

    back = False

    while back == False:
        menu.print_config()
        prompt = get_input_to_lower()

        if prompt.strip() == "back":
            back = True
            continue

        if prompt.strip() == "config":
            menu.print_config_values()
            continue

        if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")


    return