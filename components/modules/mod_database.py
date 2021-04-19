
# STANDARD LIBARIES
import csv
import os
import zipfile
from random import randint
from queue import Queue
from threading import Thread


# THIRD PARTY LIBARIES
import requests
from lxml import etree as et
import tldextract
import pandas as pd


# LOCAL LIBARIES
from config.program_config import DATA_PATH, INFO, WARNING, ERROR, ROOT_DIR, DATA_BACKUP_PATH, LEXICAL_FEATURE_LIST_COLUMN_NAMES, \
    ALEXA_FILE, BRAND_FILE, PHISHY_WORDS_FILE, LOGIN_WORDS_FILE, TLD_LOC, TLD_LOC_BACKUP, CONTENT_FEATURE_LIST_COLUMN_NAMES, \
    LEXICAL_FEATURE_DATABASE, CONTENT_FEATURE_DATABASE, SIGNATURE_FEATURE_DATABASE, SIGNATURE_FEATURE_LIST_COLUMN_NAMES
from definitions.classes.entry import Entry
from helper.logger import log
from helper.feature_helper import search_login_page
from definitions.classes import entry

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}

"""
This module contains all functions needed to work with the database files.
"""


# extract zip file
def extract_from_Zip(compressed_name, target_dir, new_name=None):
    """
    compressed_name: name of zip file
    target_dir: directory where the file has to be extracted
    new_name: new file name for extracted file
    """

    PATH = ""

    # check if file and path exists
    if not os.path.isfile(DATA_PATH + compressed_name):
        log(action_logging_enum=WARNING,
            logging_text="File [{f}] does not exist.[IN OPEN]".format(f=compressed_name))
        log(action_logging_enum=INFO, logging_text="Trying in backup folder.")

        if not os.path.isfile(DATA_BACKUP_PATH + compressed_name):
            log(action_logging_enum=ERROR,
                logging_text="File [{f}] does even not exist in backup folder.".format(
                    f=compressed_name))
            return None
        else:
            PATH = DATA_BACKUP_PATH
            log(action_logging_enum=INFO, logging_text="Found in backup folder.")
    else:
        PATH = DATA_PATH

    filename = ""

    # extract file
    if not new_name == None:
        zipdata = zipfile.ZipFile(PATH + compressed_name)
        zipinfos = zipdata.infolist()
        for zipinfo in zipinfos:
            filename = zipinfo.filename

        with zipfile.ZipFile(PATH + compressed_name, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        os.rename(target_dir + filename, target_dir + new_name)
        log(action_logging_enum=INFO, logging_text=str(
            "Extraction complete from file {c} into directory {u}.".format(c=compressed_name,u=str(target_dir + new_name))))
        return

def crawl_list_login_page(data, selenium_analysis=False, number_threads=10):
    """
    search for login pages of data set using bing and beautiful soups
    data: data set
    selenium_analysis: include selenium for search
    number_threads: number of threads for parallelism
    """

    threads = number_threads
    modified_list = []

    count = 0
    changed = 0

    # initialize token list and count, size for overview prints
    login_token_list = get_phishy_login_brand_list(login=True)

    # search login page
    def get_login():
        while True:
            url = q.get()
            url, changed_status = search_login_page(url, login_token_list, selenium_analysis=selenium_analysis)
            modify_list(url, changed_status)
            q.task_done()

    # append to list for new page
    def modify_list(url, changed_status):
        nonlocal count
        nonlocal changed
        nonlocal modified_list

        count += 1

        if url == None:
            return

        if changed_status:
            changed += 1

        if count % 10 == 0:
            log(action_logging_enum=INFO, logging_text="Found {} login pages out of {} tries.".format(changed, count))

        e = Entry(label="Bening", url=url)
        modified_list.append(e)

        return


    # do search for every thread
    try:
        q = Queue(threads * 2)
        for i in range(threads):
            t = Thread(target=get_login)
            t.daemon = True
            t.start()

        for item in data:
            q.put(item.url)
        q.join()
    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")
        return modified_list

    return modified_list


def check_status_of_website(data):
    """
    check if websites in data set can be reached (are up)
    data: set set with urls
    """

    threads = 50
    modified_list = []

    size_before = len(data)

    complete = 0
    failed = 0

    # check status for one page
    def check_status():

        nonlocal complete
        nonlocal failed
        nonlocal size_before

        while True:
            entry = q.get()
            url = entry.url
            is_up = False
            try:
                response = requests.get(url, timeout=10, headers=headers)
                response.raise_for_status()

                if response.status_code >= 200 and response.status_code < 400:
                    is_up = True
            except Exception as e:
                log(action_logging_enum=ERROR, logging_text=str(e))
                is_up = False

            if is_up:
                complete += 1
                log(action_logging_enum=INFO, logging_text="Found {} of {}. (Failed: {})".format(complete, size_before, failed))
                modify_list(entry)
            else:
                failed += 1

            q.task_done()

    # append entry to new list if up
    def modify_list(entry):
        nonlocal modified_list

        modified_list.append(entry)

        return

    # check status using 50 threads
    try:
        q = Queue(threads * 2)
        for i in range(threads):
            t = Thread(target=check_status)
            t.daemon = True
            t.start()

        for entry in data:
            q.put(entry)
        q.join()
    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")

    size_after = len(modified_list)
    log(action_logging_enum=INFO, logging_text="{} websites of {} could be reached.".format(size_after, size_before))

    return modified_list


def create_brand_list_by_alexa():
    alexa_list = open_dataset_XML_file(filename=ALEXA_FILE, iterateable="entry", label="Benign", url_label="url", label_label="label")

    brand_list = []

    for entry in alexa_list:
        url = entry.url

        brand_list.append(tldextract.extract(url).domain)

    root = et.Element("data")
    tree = et.ElementTree(root)

    for i in range(len(brand_list)):
        entry = et.SubElement(root, "entry")
        entrylabel = et.SubElement(entry, "brandname")
        entrylabel.text = str(brand_list[i])

    tree.write(open(DATA_PATH + BRAND_FILE, 'wb'), pretty_print=True)
    log(action_logging_enum=INFO,
        logging_text="Write process to XML finished for [{f}].".format(f=BRAND_FILE))



def download_file(url, filename):
    """
    download database file from url and save by filename
    url: url where data has to be downloaded
    filename: filename for data to be saved
    """

    response = requests.get(url, timeout=10, headers=headers)
    with open(DATA_PATH + filename, 'wb') as file:
        file.write(response.content)
    log(action_logging_enum=INFO,
        logging_text="Download completed for [{f}].".format(f=filename))


# mix lists randomly by inserting entries of list1 into list2 with random indicies
def mix_lists_randomly(list1, list2):
    """
    mix entries of list1 and list2 randomly
    """

    if list1 == None or list2 == None:
        log(action_logging_enum=WARNING, logging_text="At least one list was empty.")
        return None

    list1 = list(list1)
    list2 = list(list2)
    for i in range(len(list1)):
        index = randint(0, len(list2) - 1) % len(list2)
        list2.insert(index, list1[0])
        list1.pop(0)

    log(action_logging_enum=WARNING, logging_text="All lists are mixed.")
    return list2


def write_list_to_XML(filename, root, list1):
    """
    write list as xml file
    filename: name to be saved
    root: root object for xml
    list1: list with data (url, label)
    """

    root = et.Element(root)
    tree = et.ElementTree(root)

    for i in range(len(list1)):
        entry = et.SubElement(root, "entry")
        entrylabel = et.SubElement(entry, "label")
        entrylabel.text = str(list1[i].label)
        entryurl = et.SubElement(entry, "url")
        entryurl.text = str(list1[i].url)

    tree.write(open(DATA_PATH + filename, 'wb'), pretty_print=True)
    log(action_logging_enum=INFO,
        logging_text="Write process to XML finished for [{f}].".format(f=filename))


def open_dataset_XML_file(filename, iterateable, label=None, url_label="url", label_label=None, max_line_count=-1):
    """
    open xml file and write to list
    filename: xml filename
    iterateable: node containing data
    label: label for all data
    url_label: url specifier in xml data
    label_label: label specifiert in xml data
    max_line_count: maximum entries to be written to list
    """

    PATH = ""
    if not os.path.isfile(DATA_PATH + filename):
        log(action_logging_enum=WARNING,
            logging_text="File [{f}] does not exist.".format(f=filename))
        log(action_logging_enum=INFO, logging_text="Trying in backup folder.")

        if not os.path.isfile(DATA_BACKUP_PATH + filename):
            log(action_logging_enum=ERROR,
                logging_text="File [{f}] does even not exist in backup folder.".format(
                    f=filename))
            return None
        else:
            PATH = DATA_BACKUP_PATH
            log(action_logging_enum=INFO, logging_text="Found in backup folder.")
    else:
        PATH = DATA_PATH

    datalist = []
    parser = et.XMLParser(strip_cdata=False)
    xtree = et.parse(PATH + filename, parser=parser)
    root = xtree.getroot()
    index = 1

    for entry in root.iter(iterateable):
        url = entry.find(url_label).text
        e = Entry(entry.find(label_label).text, url) if not (label_label == None) else Entry(label, url)
        datalist.append(e)

        if index == max_line_count:
            break

        index += 1

    log(action_logging_enum=INFO,
        logging_text="XML File filled in list. FILE: [{f}].".format(f=filename))
    return datalist


def open_dataset_CSV_file(filename, pos_url, label="", pos_label=-1, max_line_count=-1, pass_first_line=True):
    """
    open csv data as list
    filename: csv filename
    pos_url: column number of url
    label: set label for all data
    pos_label: column number of labels
    max_line_count: maximum entries to be filled to list
    """

    datalist = []
    PATH = ""

    if not os.path.isfile(DATA_PATH + filename):
        log(action_logging_enum=WARNING,
            logging_text="File [{f}] does not exist.[IN OPEN]".format(f=filename))
        log(action_logging_enum=INFO, logging_text="Trying in backup folder.")

        if not os.path.isfile(DATA_BACKUP_PATH + filename):
            log(action_logging_enum=ERROR,
                logging_text="File [{f}] does even not exist in backup folder.".format(
                    f=filename))
            return None
        else:
            PATH = DATA_BACKUP_PATH
            log(action_logging_enum=INFO, logging_text="Found in backup folder.")
    else:
        PATH = DATA_PATH

    with open(PATH + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            line_count += 1

            if line_count <= max_line_count or max_line_count == -1:

                if pos_label != -1:
                    label = row[pos_label]
                    if str(label) == "1":
                        label = "Phish"
                    if str(label) == "0":
                        label = "Benign"

                e = Entry(label=label, url=row[pos_url])
                datalist.append(e)

            else:
                break

    log(action_logging_enum=INFO,
        logging_text="CSV File filled in list. FILE: [{f}]".format(f=filename))

    if len(datalist) > 0: datalist.pop(0)
    return datalist


# move file from data file to backup folder for downloaded files
def move_file(filename, dst_folder="", src_folder=""):
    """
    move file with filename from src_folder to dst_folder
    """

    dst_path = ROOT_DIR + dst_folder + filename if not (dst_folder == "") else DATA_BACKUP_PATH + filename
    src_path = ROOT_DIR + src_folder + filename if not (src_folder == "") else DATA_PATH + filename

    if not os.path.isfile(src_path):
        log(action_logging_enum=INFO, logging_text="File to be moved does not exist.")
        return

    if os.path.isfile(dst_path):
        os.replace(src_path, dst_path)
    else:
        os.rename(src_path, dst_path)

    log(action_logging_enum=INFO,
        logging_text="File moved from [{s}] to [{d}].".format(s=src_path, d=dst_path))


def delete_data(filename):
    """
    delete file for param filename in data_path
    """
    if os.path.isfile(DATA_PATH + filename):
        os.remove(DATA_PATH + filename)
        log(action_logging_enum=INFO,
            logging_text="File [{f}] deleted.".format(f=DATA_PATH + filename))
    else:
        log(action_logging_enum=WARNING,
            logging_text="File [{f}] dose not exists.".format(f=filename))


# write all features in one file
def write_lexical_features_CSV(feature_list):
    """
    write features for lexical filter to csv
    """

    file_name = LEXICAL_FEATURE_DATABASE
    with open(DATA_PATH + file_name, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(LEXICAL_FEATURE_LIST_COLUMN_NAMES)
        id = 1

        for entry in feature_list:
            writer.writerow([id, entry.bool_ip_netloc, entry.int_length_url, entry.bool_redirect_url, entry.bool_at_symbol_netloc,
                             entry.bool_prefix_suffix_netloc, entry.bool_subdomain_netloc, entry.int_subdomain_netloc, entry.bool_https_protocol_netloc,
                             entry.bool_other_ports_netloc, entry.bool_https_token_url, entry.int_redirect_url, entry.float_cap_noncap_letters_url,
                             entry.int_dots_url, entry.int_length_netloc,
                             entry.int_dash_netloc, entry.int_domain_tokens_netloc, entry.int_digits_netloc, entry.int_digits_path,
                             entry.int_phishy_tokens_netloc, entry.int_phishy_tokens_path, entry.bool_brand_subdomain,
                             entry.bool_brand_path, entry.bool_query_url, entry.int_query_values_url, entry.int_dots_netloc, entry.int_underscores_netloc,
                             entry.bool_validate_tld_url, entry.int_slash_path, entry.int_comma_url, entry.int_star_url, entry.int_semicolon_url,
                             entry.int_plus_url, entry.bool_javascript_url, entry.int_equals_url, entry.int_dash_url, entry.bool_fragment_url, entry.int_fragment_values_url,
                             entry.int_ampersand_url, entry.bool_html_url, entry.int_tilde_url, entry.int_symbols_url, entry.float_entropy_url,
                             entry.float_vowel_consonant_url, entry.bool_digits_netloc, entry.float_digits_letters_url, entry.int_dash_path,
                             entry.bool_domain_restlive_host, entry.bool_created_shortly_host, entry.float_percent_netloc_url,
                             entry.float_percent_path_url, entry.float_percent_query_url, entry.float_percent_fragment_url,
                             entry.float_divergence_url, entry.bool_shortening_url, entry.label, entry.url, entry.final_url])

            id += 1

    log(action_logging_enum=INFO,
        logging_text="Feature list written to CSV file. [{f}]".format(f=file_name))

# split all feature databases to validation set (3.000 entries) and training set (12.000 entries)
def generate_validation_sets():
    """
    combine the threee datasets of each filter into one
    """

    # read data
    lexical = pd.read_csv(DATA_PATH + LEXICAL_FEATURE_DATABASE)
    content = pd.read_csv(DATA_PATH + CONTENT_FEATURE_DATABASE)
    signature = pd.read_csv(DATA_PATH + SIGNATURE_FEATURE_DATABASE)

    # get url column to list
    url_lexical = lexical["URL"].tolist()
    url_content = content["URL"].tolist()
    url_signature = signature["URL"].tolist()

    # initialize indicies to be droped
    drop_lex = []
    drop_con = []
    drop_sign = []

    # initialize dataframes to be saved
    df_lex = pd.DataFrame(columns=LEXICAL_FEATURE_LIST_COLUMN_NAMES)
    df_con = pd.DataFrame(columns=CONTENT_FEATURE_LIST_COLUMN_NAMES)
    df_sign = pd.DataFrame(columns=SIGNATURE_FEATURE_LIST_COLUMN_NAMES)

    # preprocessing of urls (remove spaces)
    for i in range(len(url_content)):
        url_content[i] = url_content[i].strip()

    for i in range(len(url_signature)):
        url_signature[i] = url_signature[i].strip()

    for i in range(len(url_lexical)):
        url_lexical[i] = url_lexical[i].strip()

    processed = 0

    # search feature for url from each dataset and combine to one
    for i in range(len(url_lexical)):
        url = url_lexical[i]

        if processed == 15000:
            break

        if url not in url_signature and url not in url_content:
            if url.startswith("https://www."):
                url = url.replace("https://www.", "https://")

            if url.startswith("http://www."):
                url = url.replace("http://www.", "http://")

        print(url in url_signature, url in url_content)
        if url in url_signature and url in url_content:

            processed += 1
            log(INFO, "Processed {}/3000.".format(processed))
            ind_con = url_content.index(url)
            ind_sign = url_signature.index(url)

            df_lex = df_lex.append(lexical.iloc[i], ignore_index=True)
            df_con = df_con.append(content.iloc[ind_con], ignore_index=True)
            df_sign = df_sign.append(signature.iloc[ind_sign], ignore_index=True)

            drop_lex.append(i)
            drop_con.append(ind_con)
            drop_sign.append(ind_sign)

        else:
            print(url)

    df_lex = df_lex.drop(["ID"], axis=1)
    df_con = df_con.drop(["ID"], axis=1)
    df_sign = df_sign.drop(["ID"], axis=1)

    df_lex.to_csv(DATA_BACKUP_PATH + "val_lexical.csv", index_label="ID")
    df_con.to_csv(DATA_BACKUP_PATH + "val_content.csv", index_label="ID")
    df_sign.to_csv(DATA_BACKUP_PATH + "val_signature.csv", index_label="ID")

    lexical = lexical.drop(drop_lex)
    content = content.drop(drop_con)
    signature = signature.drop(drop_sign)

    lexical = lexical.reset_index(drop=True)
    content = content.reset_index(drop=True)
    signature = signature.reset_index(drop=True)

    for i in range(len(lexical)-12000):
        lexical = lexical.drop(0)
        lexical = lexical.reset_index(drop=True)

    for i in range(len(content)-12000):
        content = content.drop(0)
        content = content.reset_index(drop=True)

    for i in range(len(signature)-12000):
        signature = signature.drop(0)
        signature = signature.reset_index(drop=True)

    lexical = lexical.drop(["ID"], axis=1)
    content = content.drop(["ID"], axis=1)
    signature = signature.drop(["ID"], axis=1)

    lexical.to_csv(DATA_BACKUP_PATH + LEXICAL_FEATURE_DATABASE, index_label="ID")
    content.to_csv(DATA_BACKUP_PATH + CONTENT_FEATURE_DATABASE, index_label="ID")
    signature.to_csv(DATA_BACKUP_PATH + SIGNATURE_FEATURE_DATABASE, index_label="ID")
    log(INFO, "Generating validation sets completed. All sets saved in data/data_backup.")


# write all features in one file
def write_content_features_CSV(feature_list, file_name="", append=False, new_index=1):
    """
    write features for content filter to csv
    """

    if not file_name:
        file_name = CONTENT_FEATURE_DATABASE

    id = new_index

    if not append:
        with open(DATA_PATH + file_name, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(CONTENT_FEATURE_LIST_COLUMN_NAMES)

            for entry in feature_list:
                if not entry == None:
                    writer.writerow([id, entry.bool_redirect_website, entry.bool_favicon_website, entry.bool_content_extern_website, entry.int_links_extern_website, entry.bool_custom_statusbar_website,
                        entry.bool_disable_rightclick_website, entry.bool_popup_website, entry.bool_iframe_website, entry.bool_action_website, entry.bool_action_extern_website,
                        entry.bool_form_post_website, entry.int_phishy_tokens_website, entry.bool_input_website, entry.float_description_sim_website, entry.bool_bond_status_website,
                        entry.bool_freq_domain_extern_website, entry.float_login_home_website, entry.bool_copyright_website, entry.float_copyright_sim_website,
                        entry.float_title_sim_website, entry.float_unique_links_website, entry.int_input_website,
                        entry.bool_input_login_website, entry.bool_button_website, entry.bool_meta_website, entry.bool_hidden_element_website, entry.int_option_website,
                        entry.int_select_website, entry.int_th_website, entry.int_tr_website, entry.int_table_website, entry.int_href_website, entry.int_li_website, entry.int_ul_website,
                        entry.int_ol_website, entry.int_div_website, entry.int_span_website, entry.int_article_website, entry.int_p_website, entry.int_checkbox_website,
                        entry.int_button_website, entry.int_image_website, entry.label, entry.url, entry.final_url])

                    id += 1
    else:
        with open(DATA_PATH + file_name, 'a+') as file:
            writer = csv.writer(file, delimiter=',')

            for entry in feature_list:
                if not entry == None:
                    writer.writerow([id, entry.bool_redirect_website, entry.bool_favicon_website, entry.bool_content_extern_website, entry.int_links_extern_website, entry.bool_custom_statusbar_website,
                        entry.bool_disable_rightclick_website, entry.bool_popup_website, entry.bool_iframe_website, entry.bool_action_website, entry.bool_action_extern_website,
                        entry.bool_form_post_website, entry.int_phishy_tokens_website, entry.bool_input_website, entry.float_description_sim_website, entry.bool_bond_status_website,
                        entry.bool_freq_domain_extern_website, entry.float_login_home_website, entry.bool_copyright_website, entry.float_copyright_sim_website,
                        entry.float_title_sim_website, entry.float_unique_links_website, entry.int_input_website,
                        entry.bool_input_login_website, entry.bool_button_website, entry.bool_meta_website, entry.bool_hidden_element_website, entry.int_option_website,
                        entry.int_select_website, entry.int_th_website, entry.int_tr_website, entry.int_table_website, entry.int_href_website, entry.int_li_website, entry.int_ul_website,
                        entry.int_ol_website, entry.int_div_website, entry.int_span_website, entry.int_article_website, entry.int_p_website, entry.int_checkbox_website,
                        entry.int_button_website, entry.int_image_website, entry.label, entry.url, entry.final_url])

                    id += 1


    log(action_logging_enum=INFO,
        logging_text="Feature list written to CSV file. [{f}]".format(f=file_name))
    return id


# write all features in one file
def write_signature_features_CSV(feature_list, file_name=""):
    """
    write features for signature filter to csv
    """

    if not file_name:
        file_name = SIGNATURE_FEATURE_DATABASE

    id = 0

    with open(DATA_PATH + file_name, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(SIGNATURE_FEATURE_LIST_COLUMN_NAMES)

        for entry in feature_list:
            if not entry == None:
                writer.writerow([id, entry.url, entry.final_url, entry.label, entry.cert_subject, entry.ent1, entry.ent2, entry.ent3,
                                 entry.ent4, entry.ent5, entry.term1, entry.term2, entry.term3, entry.term4,
                                 entry.term5])

                id += 1

    log(action_logging_enum=INFO,
        logging_text="Feature list written to CSV file. [{f}]".format(f=file_name))



# open XML File that contains the phishywords, entry.brandwords or loginwords
def get_phishy_login_brand_list(phishy=False, brand=False, login=False):
    data_tag = -1
    filename = -1

    if phishy == True:
        data_tag = "word"
        filename = PHISHY_WORDS_FILE

    if brand == True:
        data_tag = "brandname"
        filename = BRAND_FILE

    if login == True:
        data_tag = "word"
        filename = LOGIN_WORDS_FILE

    if data_tag == -1 or filename == -1:
        log(action_logging_enum=ERROR, logging_text="No value given for identifying the list to be given.")
        return list()

    iterateable = "entry"
    PATH = ""
    if not os.path.isfile(DATA_PATH + filename):
        log(action_logging_enum=WARNING,
            logging_text="File [{f}] does not exist.".format(f=filename))
        log(action_logging_enum=INFO, logging_text="Trying in backup folder.")

        if not os.path.isfile(DATA_BACKUP_PATH + filename):
            log(action_logging_enum=ERROR,
                logging_text="File [{f}] does even not exist in backup folder.".format(
                    f=filename))
            return None
        else:
            PATH = DATA_BACKUP_PATH
            log(action_logging_enum=INFO, logging_text="Found in backup folder.")
    else:
        PATH = DATA_PATH

    data_list = []
    parser = et.XMLParser(strip_cdata=False)
    xtree = et.parse(PATH + filename, parser=parser)
    root = xtree.getroot()

    for entry in root.iter(iterateable):
        data = entry.find(data_tag).text
        data_list.append(data)
    log(action_logging_enum=INFO,
        logging_text="XML File filled in list. FILE: [{f}].".format(f=filename))
    return data_list


def get_TLD_list():
    """
    get list containing valid TLDs by IANA
    """

    tld_list = []
    line = ""

    try:
        response = requests.get(TLD_LOC, timeout=10, headers=headers)
    except Exception as e:
        log(action_logging_enum=ERROR,
            logging_text="An error occured while trying to receive the list of all TLDs by IANA.")
        log(action_logging_enum=INFO,
            logging_text="Error description: {err}".format(err=str(e)))
        log(action_logging_enum=INFO, logging_text="Taking the TLD list out of backup ressources.")

        with open(TLD_LOC_BACKUP, 'r') as f:
            line = f.readline()
            tld_list.append(line)

        tld_list.pop(0)
        return tld_list

    for char in response.text:
        if char != "\n":
            line = line + char

        if char == "\n":
            tld_list.append(line)
            line = ""

    tld_list.pop(0)
    return tld_list