# STANDARD LIBARIES

# THIRD PARTY LIBARIES

from config.program_config import INFO, WARNING
from helper import menu
from helper.helper import get_input_to_lower

# LOCAL LIBARIES
from helper.logger import log
from phishing_filter.ml_lexical import dt, rf, svm, lr, knn, ab, xgb


"""
Main file for lexical based filter
commands on console are translated to the functionality
"""

def run():

    back = False

    while back == False:
        menu.print_lexical_models()
        prompt = get_input_to_lower()

        if prompt.strip() == "back":
            back = True
            continue

        if prompt.strip() == "rf":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_rf(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    rf.load_last_score()

                if prompt.strip() == "train":
                    rf.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    rf.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "xgb":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_xgb(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    xgb.load_last_score()

                if prompt.strip() == "train":
                    xgb.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    xgb.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "ab":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_ab(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    ab.load_last_score()

                if prompt.strip() == "train":
                    ab.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    ab.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "dt":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_dt(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip().lower() == "score":
                    dt.load_last_score()

                if prompt.strip().lower() == "train":
                    dt.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    dt.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "knn":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_knn(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    knn.load_last_score()

                if prompt.strip() == "train":
                    knn.train_model()
                    continue

                if prompt.strip().startswith("predict"):
                    knn.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "lr":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_lr(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    lr.load_last_score()

                if prompt.strip() == "train":
                    lr.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    lr.predict_url(prompt.strip().split(" ")[1])
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")
            continue

        if prompt.strip() == "svm":
            back_2 = False
            optimize = False
            while back_2 == False:
                menu.print_svm(optimize)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_2 = True
                    continue

                if prompt.startswith("optimize"):
                    if prompt.split(" ")[1] == "true":
                        optimize = True
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        continue

                    if prompt.split(" ")[1] == "false":
                        log(action_logging_enum=INFO, logging_text="optimize set to: " + str(optimize))
                        optimize = False
                    continue

                if prompt.strip() == "score":
                    svm.load_last_score()

                if prompt.strip() == "train":
                    svm.train_model(do_optimize=optimize)
                    continue

                if prompt.strip().startswith("predict"):
                    svm.predict_url(prompt.strip().split(" ")[1])
                    continue

            if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")

    return