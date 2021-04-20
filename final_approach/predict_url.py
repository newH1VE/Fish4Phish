# THIRD PARTY LIBARIES
import pandas as pd

# LOCAL LIBARIES
from phishing_filter.blacklist import blacklist as bl
from helper.feature_helper import get_url_components
from config.program_config import INFO, ERROR, WARNING
from helper.logger import log
from phishing_filter.ml_lexical import rf as lex_rf
from phishing_filter.ml_content import rf as con_rf
from phishing_filter.website_signature.signature_check import SignaturClassifier
from components.modules.mod_feature_extraction import extract_features_from_signature
from phishing_filter.score_fusion import dt
from definitions.classes import blacklist_entry
from datetime import datetime, date, timedelta


def predict_url(url):
    """
    predict url using the multifilter approach with two random forests and decision tree for score fusion
    """

    log(INFO, "Starting analysing URL using the multi filter approach. [{}]".format(url))

    components = get_url_components(url)

    # check for entry in blacklist
    domainname = "{}.{}".format(components[3], components[4])
    result_domain, not_after = bl.check_for_entry(domainname=domainname)


    if result_domain is not None:
        log(INFO, "{} is listed in the blacklist until: {}".format(domainname, not_after))
        return


    # further check using all 3 filters and score fusion
    result_lex = lex_rf.predict_url(url, True)
    result_con = con_rf.predict_url(url, True)
    sig_classifier = SignaturClassifier(num_res=10)
    sig_entry = pd.Dataframe(extract_features_from_signature(url, "PREDICT"))
    result_sig = sig_classifier.predict(sig_entry.drop(["Label"], axis=1), sig_entry["Label"])

    scores = {
        "Score 1": [result_lex],
        "Score 2": [result_con],
        "Score 3": [result_sig]
    }

    columns = ["Score 1", "Score 2", "Score 3"]
    fusion_df = pd.Dataframe(scores, columns=columns)

    # get final classification by random forest
    final_classification = dt.predict(fusion_df)
    classif_str = "Benign"

    if final_classification == 1:
        classif_str = "Phishing"

        # add to blacklist
        today = date.today().strftime("%d/%m/%Y")
        not_after = datetime.strptime(today, "%d/%m/%Y") + timedelta(days=5)
        not_after_str = not_after.strftime("%d/%m/%Y")
        bl.add_entry(blacklist_entry.BlacklistEntry(domainname=domainname, not_after=not_after_str))


    log("INFO", "Final classification: {}".format(classif_str))
