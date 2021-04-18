# STANDARD LIBARIES

# THIRD PARTY LIBARIES

# LOCAL LIBARIES
from helper.logger import log
from config.program_config import INFO, WARNING
from helper.helper import get_input_to_lower
from components import comp_feature_extraction, comp_feature_selection
from helper import menu


def run():

    back = False
    signature = False
    content = False
    lexical = False

    while back == False:
        menu.print_features()
        prompt = get_input_to_lower()

        if prompt.strip() == "back":
            back = True
            continue

        if prompt.strip() == "extract":

            val_sets = False
            back_ex = False
            while back_ex == False:
                menu.print_feature_extract(signature=signature, content=content, lexical=lexical, val_sets=val_sets)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_ex = True
                    continue

                if prompt.strip() == "set signature true":
                    signature = True
                    log(action_logging_enum=INFO, logging_text="Signature extraction set to: True")
                    continue

                if prompt.strip() == "set signature false":
                    signature = False
                    log(action_logging_enum=INFO, logging_text="Signature extraction set to: False")
                    continue

                if prompt.strip() == "set content true":
                    content = True
                    log(action_logging_enum=INFO, logging_text="Content extraction set to: True")
                    continue

                if prompt.strip() == "set content false":
                    content = False
                    log(action_logging_enum=INFO, logging_text="Content extraction set to: False")
                    continue

                if prompt.strip() == "set lexical true":
                    lexical = True
                    log(action_logging_enum=INFO, logging_text="Lexical extraction set to: True")
                    continue

                if prompt.strip() == "generate val-sets":
                    val_sets = not(val_sets)
                    log(action_logging_enum=INFO, logging_text="generate validation sets set to: {}".format(val_sets))
                    continue

                if prompt.strip() == "set lexical false":
                    lexical = False
                    log(action_logging_enum=INFO, logging_text="Lexical extraction set to: False")
                    continue

                if prompt.strip() == "run":
                    comp_feature_extraction.run(content=content, lexical=lexical, signature=signature, val_sets=val_sets)
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")

        if prompt.strip() == "select":
            lexical = False
            content = False
            back_se = False
            while back_se == False:
                menu.print_feature_select(content=content, lexical=lexical)
                prompt = get_input_to_lower()

                if prompt.strip() == "back":
                    back_se = True
                    continue

                if prompt.strip() == "select content true":
                    content = True
                    log(action_logging_enum=INFO, logging_text="Content selection set to: True")
                    continue

                if prompt.strip() == "select content false":
                    content = False
                    log(action_logging_enum=INFO, logging_text="Content selection set to: False")
                    continue

                if prompt.strip() == "select lexical true":
                    lexical = True
                    log(action_logging_enum=INFO, logging_text="Lexical selection set to: True")
                    continue

                if prompt.strip() == "select lexical false":
                    lexical = False
                    log(action_logging_enum=INFO, logging_text="Lexical selection set to: False")
                    continue

                if prompt.strip() == "run":
                    comp_feature_selection.run(content=content, lexical=lexical)
                    continue

                if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")

        if prompt.strip() != "": log(action_logging_enum=WARNING, logging_text="NO COMMAND DETECTED")

    return