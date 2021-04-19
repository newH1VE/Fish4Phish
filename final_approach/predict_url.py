
from phishing_filter.blacklist import blacklist as bl
from helper.feature_helper import get_url_components
from config.program_config import INFO, ERROR, WARNING
from helper.logger import log
from phishing_filter.ml_lexical import rf as lex_rf
from phishing_filter.ml_content import rf as con_rf
from phishing_filter.website_signature.signature_check import SignaturClassifier
from components.modules.mod_feature_extraction import extract_features_from_signature
from phishing_filter.score_fusion import dt


def predict_url(url):

    log(INFO, "Starting analysing URL using the multi filter approach. [{}]".format(url))

    components = get_url_components(url)

    # check for entry in blacklist
    domainname = "{}.{}".format(components[3], components[4])
    domainname, not_after = bl.check_for_entry(domainname=domainname)


    if result is not None:
        log(INFO, "{} is listed in the blacklist until: {}".format(domainname, not_after))
        return 1


    # further check using all 3 filters and score fusion
    result_lex = lex_rf.predict_url(url, True)
    result_con = con_rf.predict_url(url, True)
    sig_classifier = SignaturClassifier(num_res=10)
    sig_entry = pd.Dataframe(extract_features_from_signature(url, "PREDICT"))
    result_sig = sig_classifier.predict(sig_entry.drop(["Label"], axis=1), sig_entry["Label"])

    score = {
        "Score 1": [result_lex],
        "Score 2": [result_con],
        "Score 3": [result_sig]
    }

    columns = ["Score 1", "Score 2", "Score 3"]

    fusion_df = pd.Dataframe(scores, columns=columns)
    final_classification = dt.predict(fusion_df)
    classif_str = "Phishing"

    if final_classification == 0:
        classif_str = "Benign"


    log("INFO", "Final classification: {}".format(classif_str))





