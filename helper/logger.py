# STANDARD LIBARIES
import os
from datetime import datetime
import inspect

# LOCAL LIBARIES
from config.program_config import LOGGING_PATH, LOGGING_TIME_FORMAT, LOGGING_FILE_NAME_FORMAT, INFO, WARNING, ERROR, \
    LOGGING_PERIOD
from definitions.classes.logcolor import LogColors


# define procedure for logging_enum
def log(action_logging_enum, logging_text, run_start=False, MODULE_NAME=None, run_complete=False):
    if action_logging_enum == INFO:
        log_print_write(action_logging_enum, logging_text, run_start, MODULE_NAME, run_complete)
    if action_logging_enum == WARNING:
        log_print_write(action_logging_enum, logging_text, run_start, MODULE_NAME, run_complete)
    if action_logging_enum == ERROR:
        log_print_write(action_logging_enum, logging_text, run_start, MODULE_NAME, run_complete)


# simple function to log module start/complete
def log_module_start(MODULE_NAME):
    log(action_logging_enum=INFO, logging_text=None, run_start=True, MODULE_NAME=MODULE_NAME)


# simple function to log module start/complete
def log_module_complete(MODULE_NAME):
    log(action_logging_enum=INFO, logging_text=None, MODULE_NAME=MODULE_NAME, run_complete=True)


# simple function to log start/end of program execution
def log_system_run():
    log_print_write(action_logging_enum=INFO, logging_text="[+++ SYSTEM RUN +++]\n")


# just printing in console not writing to file
def log_print(action_loggin_enum, logging_text, run_start=False, MODULE_NAME=None, run_complete=False):
    now = datetime.now()
    logging_time = now.strftime(LOGGING_TIME_FORMAT)
    logged_func = get_daddy()

    color = LogColors.OKGREEN
    endcolor = LogColors.ENDC

    if action_loggin_enum == WARNING:
        color = LogColors.WARNING

    if action_loggin_enum == ERROR:
        color = LogColors.FAIL

    if logged_func == "<module>":
        logged_func = "SYSTEM"

    if run_start == True and not MODULE_NAME == None:
        print("\n" + logging_time + " [+++ NEW SYSTEM RUN INITIATED FOR MODULE " + MODULE_NAME + " +++]\n")
        return

    if run_complete == True and not MODULE_NAME == None:
        print(logging_time + " [+++ SYSTEM RUN COMPLETED FOR MODULE " + MODULE_NAME + " +++]\n")
        return

    print(
        color + "[SYSTEM] | " + action_loggin_enum.get_type() + " | " + logging_time + " | " + "[Function " + logged_func + "] " + logging_text + "\n" + endcolor)


# writing to file not printing in console | delete old log files
def log_write(action_logging_enum, logging_text, run_start=False, MODULE_NAME=None, run_complete=False):
    logging_time = str(datetime.now().strftime(LOGGING_TIME_FORMAT))
    logging_file_path = LOGGING_PATH + str(datetime.now().strftime(LOGGING_FILE_NAME_FORMAT)) + ".txt"
    logged_func = get_daddy()

    color = LogColors.OKGREEN
    endcolor = LogColors.ENDC

    if logged_func == "<module>":
        logged_func = "SYSTEM"

    if run_start == False and MODULE_NAME == None and run_complete == False:
        logging_text = color + "[SYSTEM] | " + action_logging_enum.get_type() + " | " + logging_time + " | " + " | " + "[Function " + logged_func + "] " + logging_text + endcolor + "\n"

    if run_complete == True and not MODULE_NAME == None:
        logging_text = "\n\n" + "[+++ SYSTEM RUN COMPLETED FOR MODULE " + MODULE_NAME + " +++]\n\n"

    if run_start == True and not MODULE_NAME == None:
        logging_text = "\n\n" + "[+++ NEW SYSTEM RUN INITIATED FOR MODULE " + MODULE_NAME + " +++]\n\n"
        update_list = updateLogs()
        for entry in update_list:
            log(INFO, "[LOGGING SYSTEM]: File in Logs deleted. [{f}]".format(f=entry))

        if len(update_list) > 0:
            log(INFO, "[LOGGING SYSTEM]: Files in Logs updated.")

    action = 'a+'

    if run_start == True and not MODULE_NAME == None:
        logging_text = "\n\n" + "[+++ NEW SYSTEM RUN INITIATED FOR MODULE " + MODULE_NAME + " +++]\n\n"

    if not os.path.isfile(logging_file_path):
        action = 'w+'

    with open(logging_file_path, action) as log_file_obj:
        text = logging_time + logging_text
        log_file_obj.write(text)
        log_file_obj.close()


# writing to file and printing in console
def log_print_write(action_logging_enum, logging_text, run_start=False, MODULE_NAME=None, run_complete=False):
    log_print(action_logging_enum, logging_text, run_start, MODULE_NAME, run_complete)
    log_write(action_logging_enum, logging_text, run_start, MODULE_NAME, run_complete)


# get the name of the calling function for other function
def get_daddy():
    function_index = 0

    for index, call in enumerate(inspect.stack()):
        if str(call.function).__eq__("log_write") or str(call.function).__eq__("log_print"):
            function_index = index + 1
            if str(inspect.stack()[function_index].function).__eq__("log_print_write"):
                function_index += 1
                if str(inspect.stack()[function_index].function).__eq__("log"):
                    function_index += 1

    if len(inspect.stack()) < function_index - 1:
        log(action_logging_enum=INFO, logging_text="Illegal request for inspect.stack().")
        return "illegal function query"

    return str(inspect.stack()[function_index].function)


# remove logs that are older then the defined range of time including the current day as 1
def updateLogs():
    update_list = []

    if LOGGING_PERIOD == None or LOGGING_PERIOD == 0:
        return

    today = datetime.now()

    for logging_file in os.listdir(LOGGING_PATH):

        if not logging_file.__contains__('.txt'):
            continue

        file_name = logging_file.split('.')[0]
        file_date = datetime.strptime(file_name, LOGGING_FILE_NAME_FORMAT)

        if (today - file_date).days > (LOGGING_PERIOD - 1):
            if os.path.isfile(LOGGING_PATH + logging_file):
                os.remove(LOGGING_PATH + logging_file)
                update_list.append(logging_file)

    return update_list
