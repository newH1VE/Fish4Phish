
# LOCAL LIBARIES
from components.modules import mod_database as db
from helper.logger import log_module_complete, log_module_start
from config.program_config import ALEXA_FILE, PHISHTANK_FILE, DATA_PATH, DATABASE


MODULE_NAME ="DATABASE CREATION"

def run(do_download_alexa=False, do_download_phish=False, do_query_alexa=False, check_status_phishing=False, check_status_benign=False):

    log_module_start(MODULE_NAME=MODULE_NAME)


    ################ ALEXA LIST ##################

    # download all list
    if do_download_alexa==True:
        db.download_file("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip", "alexa.csv.zip")
        db.extract_from_Zip(compressed_name="alexa.csv.zip", target_dir=DATA_PATH, new_name="alexa.csv")

    # read lists from file
    if do_download_alexa:
        alexa_list = db.open_dataset_CSV_file(filename="alexa.csv", pos_url=1, label="Benign", max_line_count=16000)
    else:
        alexa_list = db.open_dataset_XML_file(ALEXA_FILE, iterateable="entry", label="Benign", url_label="url")

    if do_query_alexa == True:
            alexa_list = db.crawl_list_login_page(data=alexa_list, selenium_analysis=False, number_threads=10)

    if do_download_alexa:
        db.delete_data("alexa.csv.zip")
        db.move_file("alexa.csv")

    ################ PHISHTANK LIST ##################

    if do_download_phish == True: db.download_file(
        "http://data.phishtank.com/data/9613df2569182bccf42c10b7c49387d62d877838ba4d87ab2ebad9b13ae9ad66/online-valid.xml",
        PHISHTANK_FILE)

    if not alexa_list == None: db.write_list_to_XML(filename=ALEXA_FILE, root="data", list1=alexa_list)

    phishtank_list = db.open_dataset_XML_file(filename=PHISHTANK_FILE, iterateable="entry", label="Phish")

    if check_status_phishing:
        phishtank_list = db.check_status_of_website(phishtank_list)

    if check_status_benign:
        alexa_list = db.check_status_of_website(alexa_list)

    if len(phishtank_list) != len(alexa_list):
        if len(phishtank_list) > len(alexa_list):
            diff = len(phishtank_list) - len(alexa_list)

            for i in range(diff):
                phishtank_list.pop(0)
        else:
            diff = len(alexa_list) - len(phishtank_list)

            for i in range(diff):
                alexa_list.pop(0)

    if not phishtank_list == None: db.write_list_to_XML(filename=PHISHTANK_FILE, root="data", list1=phishtank_list)

    # kaggle database available at: https://www.kaggle.com/kunal4892/phishingandlegitimateurls
    # kaggle_list = db.openCSVFile(filename="kaggle.csv", pos_url=0, pos_label=11)
    #db.deleteData("kaggle.csv")
    # if not kaggle_list == None: db.writeListtoXML(filename=KAGGLE_FILE, root="data", list=kaggle_list)


    ################ FINAL LIST ##################

    # create mox of kaggle, phishtank and alexa
    final_list = db.mix_lists_randomly(alexa_list, phishtank_list)

    # safe final list
    if not final_list == None: db.write_list_to_XML(filename=DATABASE, root="data", list1=final_list)

    log_module_complete(MODULE_NAME=MODULE_NAME)