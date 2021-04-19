from configparser import ConfigParser

from helper.logger import log
from definitions.enum_def.logging import  logging_enum
from config.program_config import CONFIG_FILE

file = CONFIG_FILE
config = ConfigParser()
config.read(file)

"""
This configuration creates/modifies the fish4phish.ini in DATA_PATH
new configurations can be saved and older ones modified
"""

def get_element(section, element):
    return str(config[section][element])

def set_element(section, element, value):
    config.set(section, element, str(value))
    log(action_logging_enum=logging_enum.INFO, logging_text="{} of section {} set to {}".format(element, section, str(value)))
    write_config()

def check_for_section(section):
    if config.has_section(section):
        return True

    return False

def check_for_element(section, element):
    if check_for_section(section):
        if config.has_option(section, option=element):
            return True
        else:
            return False
    else:
        return False

def add_section(section):
    if not config.has_section(section):
        config.add_section(section)
        log(action_logging_enum=logging_enum.INFO, logging_text="Added section {} to config.".format(section))
        write_config()

def add_element(section, element, value):
    if not config.has_section(section):
        add_section(section)

    config.set(section, element, str(value))
    write_config()

def write_config():
    with open(file, 'w') as configfile:
        config.write(configfile)