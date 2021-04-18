
# STANDARD LIBARIES

# THIRD PARTY LIBARIES
import pandas as pd

# LOCAL LIBARIES
from components.modules.mod_feature_selection import do_greedy_search
from config.program_config import LEXICAL_FEATURE_DATABASE, CONTENT_FEATURE_DATABASE, DATA_PATH
from helper.logger import log_module_complete, log_module_start

MODULE_NAME="FEATURE SELECTION"

def run(content, lexical):

    if lexical:
        data = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)

        data = data.drop(["ID", "URL", "Final URL", "Has Shortening", "Number Comma", "Number Stars" , "Number Plus" ,
                          "Has HTTPS Token", "Has Javascript", "Has At Symbol", "Cert Restlive",
                          "Number PhishyTokens Netloc", "Number Underscore Netloc", "Has Fragment", "Number Tilde"
                          , "Number Dash Path", "Has Brand Subdomain", "Has Brand Path", "Has HTML Code",
                          "Number Subdomains", "Has Valide TLD", "Has Subdomains", "Has IP", "Has Other Port",
                          "Has Redirect", "Number Fragment Values", "Ratio Fragment/URL", "Cert Created Shortly",
                          "Number Semicolon", "Number Ampersand"], axis=1)

        result = do_greedy_search(data_set=data, lexical=lexical)

    if content:
        data = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)

        data = data.drop(["ID", "URL", "Final URL", "Number Checkbox", "Number Select", "Number OL", "Number TR",
                        "Number TH", "Number Article", "Number Table", "Has Form with POST", "Has Extern Action", "Has Input for Login",
                        "Has PopUp", "Has Favicon", "Has Custom StatusBar", "Has Disabled RightClick"], axis=1)
        print(data.columns)
        result = do_greedy_search(data_set=data, content=content)



    # module complete
    log_module_complete(MODULE_NAME=MODULE_NAME)