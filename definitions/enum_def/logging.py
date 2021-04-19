
# STANDARD LIBARIES
import enum

# differnet types of logging

class logging_enum(enum.Enum):
    INFO=0
    WARNING=1
    ERROR=2

    def get_type(self):
        if self == logging_enum.INFO:
            return "[INFO] "
        if self == logging_enum.WARNING:
            return "[WARN] "
        if self == logging_enum.ERROR:
            return "[ERR] "

        return "[FATAL ERROR IN LOGGING] "