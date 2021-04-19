# STANDARD LIBARIES
import linecache
import re
import sys
import time
from collections import Counter
from threading import Thread
from urllib.parse import urljoin

# THIRD PARTY LIBARIES
import bs4
import favicon
import html_similarity
import pandas as pd
import ray
from datetime import datetime
from queue import Queue
from validators import url as validate_url

# LOCAL LIBARIES
from components.modules.mod_database import get_phishy_login_brand_list, get_TLD_list
from config.program_config import INFO, ERROR, WARNING, LEXICAL_FEATURE_LIST_COLUMN_NAMES, \
    CONTENT_FEATURE_LIST_COLUMN_NAMES, SIGNATURE_FEATURE_LIST_COLUMN_NAMES
from definitions.classes.feature_entry_content import FeatureEntryContent
from definitions.classes.feature_entry_lexical import FeatureEntryLexical
from definitions.classes.redirect_entry import RedirectEntry
from definitions.classes.signature_entry import SignatureEntry
from helper.feature_helper import get_url_components, is_IP, get_redirects_list
from helper.feature_helper import request_used_protocol, entropy_of_string, get_ssl_information, \
    detect_website_language, find_extern_links, get_website_text, string_similarity, get_element_count, \
    compute_divergence, get_common_entities, get_common_terms, get_ssl_subject
from helper.helper import get_redirects
from helper.logger import log

tld_list = get_TLD_list()
brand_list = get_phishy_login_brand_list(brand=True)
phishy_list = get_phishy_login_brand_list(phishy=True)
login_list = get_phishy_login_brand_list(login=True)
correction = 0
count = 0
failed = 0


def extract_features_from_URL_list(data, predict=False, threads=50):
    """
    extract url features from list using multiple threads
    threads: number of threads
    data: list containing urls and label
    predict: extract from list for prediction (set label to: PREDICT)
    """

    df = pd.DataFrame()

    feature_list = []
    count = 0
    size = len(data)
    failed = 0

    def merge_dataframes(dataframe):
        try:
            nonlocal df

            if not pd.DataFrame(dataframe).empty:

                if df.empty:
                    df = dataframe
                else:
                    df = pd.concat([df, dataframe], ignore_index=True)
        except Exception as e:
            log(action_logging_enum=WARNING, logging_text=str(e))

    def get_feature_entry():
        while True:
            entry = q.get()

            if predict:
                new_entry = extract_features_from_URL(entry, "Predict", predict=True)
                merge_dataframes(new_entry)
            else:
                new_entry = extract_features_from_URL(entry.url, entry.label)
                append_in_list(new_entry, entry.url)

            q.task_done()

    def append_in_list(new_entry, url):
        nonlocal feature_list
        nonlocal count
        nonlocal failed

        if not new_entry == None:
            count += 1
            log(action_logging_enum=INFO,
                logging_text="Processed datapoint {} of {}. restlive: {} created: {} (Failed: {})".format(count, size,
                                                                                                          new_entry.bool_domain_restlive_host,
                                                                                                          new_entry.bool_created_shortly_host,
                                                                                                          failed))
            feature_list.append(new_entry)
        else:
            failed += 1
            log(action_logging_enum=WARNING, logging_text="Failed for {} of {}. Entry: {}".format(failed, size, url))

        return

    try:
        q = Queue(threads * 2)
        for i in range(threads):
            t = Thread(target=get_feature_entry)
            t.daemon = True
            t.start()

        for entry in data:
            q.put(entry)

        q.join()
    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")

        if predict:
            return df
        else:
            return feature_list

    if predict:
        return df
    else:
        return feature_list


def extract_features_from_URL(url, label="", predict=False):
    """
    extract all features from url, if predict set to true a pandas dataframe is created
    """

    # get components netloc, filepath, query, path, Subdomain, SLD (domainname), TLD (incl. ccTLD)
    df = pd.DataFrame()

    try:
        data = url
        orig_web = ""

        redirect_object = isinstance(data, RedirectEntry)

        if redirect_object:
            url = data.url
            int_redirect_url = data.redirects
            protocol = data.protocol
            content = data.content

            components_orig = get_url_components(url)

            domain_orig = components_orig[3]
            suffix_orig = components_orig[4]

            if suffix_orig:
                orig_web = "{}.{}".format(domain_orig, suffix_orig)
            else:
                orig_web = domain_orig

        else:

            components_orig = get_url_components(url)

            domain_orig = components_orig[3]
            suffix_orig = components_orig[4]

            if suffix_orig:
                orig_web = "{}.{}".format(domain_orig, suffix_orig)
            else:
                orig_web = domain_orig

            # number of redirects
            resp_url, int_redirect_url, protocol, content = get_redirects(url)
            final_url = resp_url

        components = get_url_components(url)

        fqdn = components[0]
        scheme = components[1]
        subdomain = components[2]
        domain = components[3]
        suffix = components[4]
        port = components[5]
        path = components[6]
        query = components[7]
        fragment = components[8]

        if suffix:
            new_web = "{}.{}".format(domain, suffix)
        else:
            new_web = domain

        # FEATURE EXTRACTION START

        netloc = fqdn
        url_no_prot = url

        if port:
            netloc = netloc + ":" + port

        if not scheme:
            scheme = "http"
            if protocol == 1:
                scheme = "https"

        url_no_prot = url.replace(scheme + "://", "", 1)

        # IP address in netloc existent
        bool_ip_netloc = is_IP(domain)

        if subdomain == "" and not bool_ip_netloc:
            if url.startswith("https://"): url = url.replace("https://", "https://www.", 1)
            if url.startswith("http://"): url = url.replace("http://", "http://www.", 1)

        # URL FEATURES

        # use of shortening service
        bool_shortening_url = False
        if not orig_web == new_web:
            bool_shortening_url = True

        # kullback-leibler divergence
        float_divergence_url = 0
        if not content == -1:
            float_divergence_url = compute_divergence(url, content)

        # length of the URL
        int_length_url = len(url)

        # uses redirect (e.g. shortening service)
        bool_redirect_url = bool(fqdn.__contains__('//'))

        # https used as token in url
        bool_https_token_url = bool(fqdn.lower().__contains__("https"))

        # ratio of capital and non-capital letters
        cap = sum(1 for c in url if c.isupper())
        non_cap = sum(1 for c in url if c.islower())
        float_cap_noncap_letters_url = float(cap / non_cap)

        # number of dots in url
        int_dots_url = url.count(".")

        # number of queries in url > custom bool has query and number of values in query
        bool_query_url = False
        int_query_values_url = 0

        if query:
            bool_query_url = True
            int_query_values_url = query.count("&") + 1

        # validate tld
        bool_validate_tld_url = True

        if not suffix == '':
            suffix_tokens = suffix.split(".")

            for token in suffix_tokens:
                if token not in tld_list:
                    bool_validate_tld_url = False
                    break

        # number of comma in url
        int_comma_url = url.count(",")

        # number of stars in url
        int_star_url = url.count("*")

        # number of semicolon in url
        int_semicolon_url = url.count(";")

        # number of spaces in url
        int_plus_url = url.count(" ")

        # javascript in url
        bool_javascript_url = bool(url.lower().__contains__("javascript:"))

        # number of equal signs in url
        int_equals_url = url.count("=")

        # number of dashes in url
        int_dash_url = url.count("-")

        # number of hash in url > custom as for query bool has fragment and number of value pairs
        bool_fragment_url = True
        int_fragment_values_url = 0
        if fragment:
            bool_fragment_url = False
            int_fragment_values_url = fragment.count("&") + 1

        # number of ampersands in url
        int_ampersand_url = url.count("&")

        # usage of % in url
        bool_html_url = bool(url.__contains__("%"))

        # number of tilde in url
        int_tilde_url = url.count("~")

        # number of not alpha-numerical symbols | without protocol ... https://
        int_symbols_url = sum((1 for c in url_no_prot if not c.isalpha() and not c.isdigit()), 1)

        # entropy of the url
        float_entropy_url = entropy_of_string(url)

        # ratio of vowel to consonant
        vowel = sum(1 for char in url.lower() if char in ('a', 'e', 'i', 'o', 'u',) and char.isalpha())
        consonant = sum(1 for char in url if char.isalpha()) - vowel
        float_vowel_consonant_url = vowel / consonant

        # ratio of numbers to letters in url
        digits = sum(1 for char in url if char.isdigit())
        letters = sum(1 for char in url if char.isalpha())
        float_digits_letters_url = digits / letters

        # brand in url | not used because of brand token list consist of alexa list domains as tokens
        # bool_brand_url = any(url.__contains__(str(brand).lower()) for brand in brand_list)

        # OWN URL FEATURES

        # percentage length of netloc to url
        float_percent_netloc_url = round(len(netloc) / len(url), 2)

        # percentage length of path to url
        float_percent_path_url = round(len(path) / len(url), 2)

        # percentage length of query to url
        float_percent_query_url = round(len(query) / len(url), 2)

        # percentage length of fragment to url
        float_percent_fragment_url = round(len(fragment) / len(url), 2)

        # NETLOC FEATURES

        # uses @ in netloc
        bool_at_symbol_netloc = bool(str(url).split(fqdn)[0].__contains__("@"))

        # prefix or suffix in netloc used
        bool_prefix_suffix_netloc = bool(netloc.__contains__('-'))

        # contains subdomains > custom number of subdomains
        bool_subdomain_netloc = False
        int_subdomain_netloc = 0
        if not components[4] == "www" and not components[4] == "":
            bool_subdomain_netloc = True
            int_subdomain_netloc = components[4].count(".") + 1

        # usage of https or not
        if protocol == -1:
            bool_https_protocol_netloc = request_used_protocol(url)
        else:
            bool_https_protocol_netloc = protocol

        # usage of abnormal ports and identify port for further features
        bool_other_ports_netloc = False

        if bool_https_protocol_netloc == 1 and not port: port = 443
        if bool_https_protocol_netloc == 1 and not port: port = 80

        if fqdn.__contains__(':'):
            port = netloc.split(':')[1]

            if not port == 80 and not port == 443:
                bool_other_ports_netloc = True

        # length of netloc
        int_length_netloc = len(netloc)

        # number of domains in netloc
        int_domains_netloc = netloc.count(".")

        # number of dashes in netloc
        int_dash_netloc = netloc.count("-")

        # count tokens in netloc that are created via - and .
        int_domain_tokens_netloc = sum((1 for c in fqdn if not c.isalpha() and not c.isdigit()), 1)

        # number of digits in netloc
        int_digits_netloc = sum(1 for c in netloc if c.isdigit())

        # number of dots in netloc
        int_dots_netloc = netloc.count(".")

        # number of underscores in netloc
        int_underscores_netloc = netloc.count("_")

        # true if digits in netloc contained
        bool_digits_netloc = bool(any(char.isdigit() for char in netloc))

        # PATH FEATURES

        # number digits in path
        int_digits_path = sum(1 for c in path if c.isdigit())

        # number of phishy words in netloc
        int_phishy_tokens_netloc = sum(1 for word in phishy_list if netloc.__contains__(word))

        # number of phishy tokens in path
        int_phishy_tokens_path = sum(1 for word in phishy_list if path.__contains__(word))

        # brand in path
        bool_brand_path = bool(any(path.lower().__contains__(str(brand).lower()) for brand in brand_list))

        # number of slashes in path
        int_slash_path = path.count("/")

        # number of dashes in path
        int_dash_path = path.count("-")

        #  SUBDOMAIN FEATURES

        # brand in subdomain
        bool_brand_subdomain = bool(any(subdomain.lower().__contains__(str(brand).lower()) for brand in brand_list))

        # HOST FEATURES

        # define todays date
        today_date = datetime.date(datetime.now())

        # domain created longer than one month ago
        # certificate based

        bool_created_shortly_host = False
        bool_domain_restlive_host = False

        creation_date = None
        expire_date = None

        if bool_https_protocol_netloc:
            expire_date, creation_date = get_ssl_information(url)

        if not creation_date == None:
            num_months = (today_date.year - creation_date.year) * 12 + (today_date.month - creation_date.month)
            if num_months < 2:
                bool_created_shortly_host = True

        # restlive of domain more than 3 months
        # all certificate based

        if not expire_date == None:

            num_months = (expire_date.year - today_date.year) * 12 + (expire_date.month - today_date.month)
            if num_months >= 3:
                bool_domain_restlive_host = True

        if not predict:
            entry = FeatureEntryLexical(bool_ip_netloc=bool_ip_netloc, int_length_url=int_length_url,
                                        bool_redirect_url=bool_redirect_url,
                                        bool_at_symbol_netloc=bool_at_symbol_netloc,
                                        bool_prefix_suffix_netloc=bool_prefix_suffix_netloc,
                                        bool_subdomain_netloc=bool_subdomain_netloc,
                                        int_subdomain_netloc=int_subdomain_netloc,
                                        bool_https_protocol_netloc=bool_https_protocol_netloc,
                                        bool_other_ports_netloc=bool_other_ports_netloc,
                                        bool_https_token_url=bool_https_token_url,
                                        int_redirect_url=int_redirect_url,
                                        float_cap_noncap_letters_url=float_cap_noncap_letters_url,
                                        int_dots_url=int_dots_url, int_length_netloc=int_length_netloc,
                                        int_domains_netloc=int_domains_netloc,
                                        int_dash_netloc=int_dash_netloc,
                                        int_domain_tokens_netloc=int_domain_tokens_netloc,
                                        int_digits_netloc=int_digits_netloc,
                                        int_digits_path=int_digits_path,
                                        int_phishy_tokens_netloc=int_phishy_tokens_netloc,
                                        int_phishy_tokens_path=int_phishy_tokens_path,
                                        bool_brand_subdomain=bool_brand_subdomain, bool_brand_path=bool_brand_path,
                                        bool_query_url=bool_query_url, int_query_values_url=int_query_values_url,
                                        int_dots_netloc=int_dots_netloc, int_underscores_netloc=int_underscores_netloc,
                                        bool_validate_tld_url=bool_validate_tld_url,
                                        int_slash_path=int_slash_path, int_comma_url=int_comma_url,
                                        int_star_url=int_star_url, int_semicolon_url=int_semicolon_url,
                                        int_plus_url=int_plus_url, bool_javascript_url=bool_javascript_url,
                                        int_equals_url=int_equals_url,
                                        int_dash_url=int_dash_url, bool_fragment_url=bool_fragment_url,
                                        int_fragment_values_url=int_fragment_values_url,
                                        int_ampersand_url=int_ampersand_url, bool_html_url=bool_html_url,
                                        int_tilde_url=int_tilde_url, int_symbols_url=int_symbols_url,
                                        float_entropy_url=float_entropy_url,
                                        float_vowel_consonant_url=float_vowel_consonant_url,
                                        bool_digits_netloc=bool_digits_netloc,
                                        float_digits_letters_url=float_digits_letters_url, int_dash_path=int_dash_path,
                                        bool_domain_restlive_host=bool_domain_restlive_host,
                                        bool_created_shortly_host=bool_created_shortly_host,
                                        float_percent_netloc_url=float_percent_netloc_url,
                                        float_percent_path_url=float_percent_path_url,
                                        float_percent_query_url=float_percent_query_url,
                                        float_percent_fragment_url=float_percent_fragment_url,
                                        float_divergence_url=float_divergence_url,
                                        bool_shortening_url=bool_shortening_url, label=label, url=url,
                                        final_url=final_url)

            return entry
        elif predict:
            data = {'ID': 0,
                    'Has IP': [bool_ip_netloc],
                    'Length URL': [int_length_url],
                    'Has Redirect': [bool_redirect_url],
                    'Has At Symbol': [bool_at_symbol_netloc],
                    'Has Token Netloc': [bool_prefix_suffix_netloc],
                    'Has Subdomains': [bool_subdomain_netloc],
                    'Number Subdomains': [int_subdomain_netloc],
                    'Has HTTPS': [bool_https_protocol_netloc],
                    'Has Other Port': [bool_other_ports_netloc],
                    'Has HTTPS Token': [bool_https_token_url],
                    'Number Redirects': [int_redirect_url],
                    'Ratio Cap/NonCap': [float_cap_noncap_letters_url],
                    'Number Dots': [int_dots_url],
                    'Length Netloc': [int_length_netloc],
                    'Number Dash Netloc': [int_dash_netloc],
                    'Number Tokens Netloc': [int_domain_tokens_netloc],
                    'Number Digits Netloc': [int_digits_netloc],
                    'Number Digits Path': [int_digits_path],
                    'Number PhishyTokens Netloc': [int_phishy_tokens_netloc],
                    'Number PhishyTokens Path': [int_phishy_tokens_path],
                    'Has Brand Subdomain': [bool_brand_subdomain],
                    'Has Brand Path': [bool_brand_path],
                    'Has Query': [bool_query_url],
                    'Number Query Parameters': [int_query_values_url],
                    'Number Dots Netloc': [int_dots_netloc],
                    'Number Underscore Netloc': [int_underscores_netloc],
                    'Has Valide TLD': [bool_validate_tld_url],
                    'Number Slash Path': [int_slash_path],
                    'Number Comma': [int_comma_url],
                    'Number Stars': [int_star_url],
                    'Number Semicolon': [int_semicolon_url],
                    'Number Plus': [int_plus_url],
                    'Has Javascript': [bool_javascript_url],
                    'Number Equals': [int_equals_url],
                    'Number Dash': [int_dash_url],
                    'Has Fragment': [bool_fragment_url],
                    'Number Fragment Values': [int_fragment_values_url],
                    'Number Ampersand': [int_ampersand_url],
                    'Has HTML Code': [bool_html_url],
                    'Number Tilde': [int_tilde_url],
                    'Number Symbols': [int_symbols_url],
                    'Entropy': [float_entropy_url],
                    'Ratio Vowel/Consonant': [float_vowel_consonant_url],
                    'Has Digits Netloc': [bool_digits_netloc],
                    'Ratio Digit/Letter': [float_digits_letters_url],
                    'Number Dash Path': [int_dash_path],
                    'Cert Restlive': [bool_domain_restlive_host],
                    'Cert Created Shortly': [bool_created_shortly_host],
                    'Ratio Netloc/URL': [float_percent_netloc_url],
                    'Ratio Path/URL': [float_percent_path_url],
                    'Ratio Query/URL': [float_percent_query_url],
                    'Ratio Fragment/URL': [float_percent_fragment_url],
                    'KL Divergence': [float_divergence_url],
                    'Has Shortening': [bool_shortening_url],
                    'Label': [label],
                    'URL': [url],
                    'Final URL': [final_url]
                    }

            columns = list(LEXICAL_FEATURE_LIST_COLUMN_NAMES)

            df = pd.DataFrame(data, columns=columns)

            return df

    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        log(ERROR, 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        log(action_logging_enum=ERROR, logging_text="Could not extract lexical features from url: {}".format(url))
        if predict:
            return df

    return None


def extract_features_from_website_list(data):
    """
    extract all content based features from list with urls
    """

    threads = 80
    feature_list = []

    count = 0
    size = len(data)
    failed = 0
    succeed = 0

    def get_feature_entry():
        nonlocal count
        while True:
            entry = q.get()
            new_entry = extract_features_from_website(entry.url, entry.label, False)
            count += 1

            if new_entry != None:
                append_in_list(new_entry, entry.url)

            q.task_done()

        return

    def append_in_list(new_entry, url):
        nonlocal feature_list
        nonlocal succeed
        nonlocal failed

        if not new_entry == None:
            succeed += 1
            log(action_logging_enum=INFO,
                logging_text="Processed datapoint {} of {} (Failed: {})".format(succeed, size, failed))
            feature_list.append(new_entry)
        else:
            failed += 1
            log(action_logging_enum=ERROR, logging_text="Failed for {} of {}. Entry: {}".format(failed, size, url))

        return

    try:
        q = Queue(threads * 2)
        for i in range(threads):
            t = Thread(target=get_feature_entry)
            t.daemon = True
            t.start()

        for entry in data:
            q.put(entry)
        q.join()
    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")
        return feature_list

    return feature_list


def extract_features_from_website(url, label, predict):
    """
        extract all features from website, if predict set to true a pandas dataframe is created
    """

    try:
        global brand_list
        global phishy_list
        global login_list
        global tld_list
        # save original url for object instance
        url_orig = url

        # get different components of url
        components = get_url_components(url)

        fqdn = components[0]
        scheme = components[1]
        subdomain = components[2]
        domain = components[3]
        suffix = components[4]
        port = components[5]
        path = components[6]
        query = components[7]
        fragment = components[8]

        netloc = fqdn
        url_no_prot = url

        if scheme:
            netloc = scheme + "://" + fqdn

            if port:
                netloc = netloc + ":" + port

            url_no_prot = url.replace(scheme + "://", "", 1)

        # check for redirects of url
        resp_url, num_redirects, protocol, content = get_redirects(url)

        # try again if no connection could have been established
        if content == -1:
            time.sleep(3)
            resp_url, num_redirects, protocol, content = get_redirects(url)

            if content == -1:
                return None

        # get content for homepage
        hp_url, hp_num_redirects, hp_protocol, hp_content = get_redirects(
            "{}://www.{}.{}".format(scheme, domain, suffix))

        if hp_content == -1:
            time.sleep(3)
            hp_url, hp_num_redirects, hp_protocol, hp_content = get_redirects(
                "{}://www.{}.{}".format(scheme, domain, suffix))

        # read content in parser
        if not hp_content == -1:
            hp_soup = bs4.BeautifulSoup(hp_content.lower(), 'html.parser')

        soup = bs4.BeautifulSoup(content.lower(), 'html.parser')

        url = resp_url

        # number of redirects done by website
        if num_redirects > 0:
            bool_redirect_website = True
        else:
            bool_redirect_website = False

        # website has favicon/ check if website has favicon
        bool_favicon_website = False

        try:
            icon = favicon.get(url, timeout=3)
            bool_favicon_website = True
        except Exception as e:
            bool_favicon_website = False

        # website has links pointing to extern content
        bool_content_extern_website = False

        # number of links pointing to extern content
        int_links_extern_website = 0
        bool_content_extern_website, int_links_extern_website = find_extern_links(content.lower(), domain, suffix, url)

        # check for custom status bar
        bool_custom_statusbar_website = bool(str(content).lower().replace(" ", "").__contains__("window.status="))

        # custom right click
        bool_disable_rightclick_website = False

        if str(content).replace(" ", "").lower().__contains__("document.oncontextmenu="):
            bool_disable_rightclick_website = True

        res = soup.findAll("body")

        if res:
            for element in res:
                try:
                    right_click_arg = element['oncontextmenu']
                    if str(right_click_arg) == "return false":
                        bool_disable_right_click = True
                except Exception as e:
                    continue

        # has pop up window
        bool_popup_website = False
        hidden_count = 0
        res = soup.findAll("div")

        if res:
            for tag in res:
                try:
                    arg = tag['class']
                    if "popup" in arg:
                        bool_popup_website = True
                except Exception as e:
                    pass
                try:
                    arg = tag['style']
                    arg = str(arg).replace(" ", "")

                    if arg.__contains__("display:none") or arg.__contains__("visibility:hidden"):
                        hidden_count += 1
                except Exception as e:
                    continue

        # has iframe
        bool_iframe_website = False
        res = soup.findAll("iframe")
        if res:
            bool_iframe_website = True

        # has action tag > custom 2. feature - is action extern?
        bool_action_website = False
        bool_action_extern_website = False

        # has bool form post
        bool_form_post_website = False

        res = soup.findAll("form")

        if res:
            for element in res:
                try:
                    if element["action"]:
                        bool_action_website = True
                        action_url = element["action"]

                        if validate_url(action_url) or validate_url(urljoin(netloc, action_url)):

                            if validate_url(urljoin(netloc, action_url)):
                                action_url = urljoin(netloc, action_url)

                            extracted_action_url = get_url_components(action_url)

                            domain_action_url = extracted_action_url[3]
                            suffix_action_url = extracted_action_url[4]

                            if not suffix == suffix_action_url or not domain == domain_action_url:
                                bool_action_extern_website = True
                                break

                    if element["method"] == "post":
                        bool_form_post_website = True
                except Exception as e:
                    continue

        # has phishy tokens in visible content
        int_phishy_tokens_website = 0

        for text in soup.stripped_strings:
            int_phishy_tokens_website += sum(1 for word in phishy_list if text.__contains__(word))

        # has input tag
        bool_input_website = False
        if get_element_count("input", soup) > 0: bool_input_website = True

        # find meta description
        res = soup.find('meta', attrs={'name': 'og:description'})
        if not res:
            res = soup.find('meta', attrs={'property': 'description'})
        if not res:
            res = soup.find('meta', attrs={'name': 'description'})

        if not hp_content == -1:
            hp_res = hp_soup.find('meta', attrs={'name': 'og:description'})
            if not hp_res:
                hp_res = hp_soup.find('meta', attrs={'property': 'description'})
            if not hp_res:
                hp_res = hp_soup.find('meta', attrs={'name': 'description'})

        float_description_sim_website = 0

        if hp_content == -1:
            float_description_sim_website = -1

        if not hp_content == -1:
            if res and hp_res:
                try:
                    hp_desc = hp_res['content']
                    desc = res['content']

                    # compute similarity of description from home and login page
                    float_description_sim_website = string_similarity(desc, hp_desc)
                except Exception:
                    pass

        # bond status login and homepage
        bool_bond_status_website = False

        # most frequent domain ist extern > tru/false
        bool_freq_domain_extern_website = False
        res = soup.findAll("a")
        domain_list = []
        link_list = []
        href_count = 0
        redirect_object_list = []

        if res:
            for a_tag in res:
                try:
                    href = a_tag.attrs.get("href")

                    href_count += 1

                    if validate_url(href) or validate_url(urljoin(netloc, href)):

                        if validate_url(urljoin(netloc, href)):
                            href = urljoin(netloc, href)

                        if href == hp_url:
                            bool_bond_status_website = True

                        components_href = get_url_components(href)

                        domain_href = components_href[3]
                        suffix_href = components_href[4]

                        if is_IP(domain):
                            continue
                        link_list.append(href)
                        domain_list.append("{},{}".format(domain_href, suffix_href))

                except Exception as e:
                    continue

            link_list = list(set(link_list))
            link_list = link_list[:10]
            if not hp_content == -1:
                try:
                    redirect_object_list = get_redirects_list(link_list)

                except Exception as e:
                    log(action_logging_enum=ERROR, logging_text=str(e))

                if redirect_object_list:
                    for redirect_object in redirect_object_list:

                        if not bool_bond_status_website and not hp_content == -1 and redirect_object_list:
                            try:
                                website_sim = html_similarity.similarity(str(hp_content).lower(),
                                                                         str(redirect_object.content).lower(), k=0.3)

                                if website_sim == 1:
                                    bool_bond_status_website = True
                            except Exception:
                                continue

        if domain_list:
            occure_count = Counter(domain_list)
            most_freq = occure_count.most_common(1)[0][0]
            most_frq_domain, most_freq_suffix = most_freq.split(",", 1)

            if not str(most_frq_domain) == domain or not str(suffix) == most_freq_suffix:
                bool_freq_domain_extern_website = True

        # jaccard similarity between homepage and login page
        float_login_home_website = 0
        if not hp_content == -1:
            try:
                float_login_home_website = html_similarity.similarity(str(content).lower(), str(hp_content).lower(),
                                                                      k=0.3)
            except Exception:
                pass
        # website has copyright
        bool_copyright_website = False

        # similarity from copyright of login page and home page
        copy = ""
        hp_copy = ""
        if not hp_content == -1:
            float_copyright_sim_website = 0
            for text in soup.stripped_strings:
                if '©' in text:
                    copy = re.sub(r'\s+', ' ', text)
                    bool_copyright_website = True

            for text in hp_soup.stripped_strings:
                if '©' in text:
                    hp_copy = re.sub(r'\s+', ' ', text)

            if copy and hp_copy:
                float_copyright_sim_website = string_similarity(copy, hp_copy)
        else:
            float_copyright_sim_website = 0

        # similarity from title of login page and home page
        float_title_sim_website = 0
        if not hp_content == -1:
            try:
                title = soup.title.text
                hp_title = hp_soup.title.text
                float_title_sim_website = string_similarity(title, hp_title)
            except Exception:
                float_title_sim_website = 0
                pass

        # unique links/all links on page
        float_unique_links_website = 0
        if link_list:
            float_unique_links_website = len(list(set(link_list))) / len(link_list)

        # lexical analysis for all links on website
        bool_link_analysis_website = True
        # dataframe = pd.DataFrame()
        # try:
        # redirect_object = RedirectEntry(url=url, redirects=num_redirects, content=content, protocol=protocol)
        # dataframe = pd.DataFrame(extract_features_from_URL(redirect_object, "Predict", brand_list=brand_list,
        # tld_list=tld_list, phishy_list=phishy_list, predict=True))
        # except Exception as e:
        # pass

        # if not dataframe.empty:
        # try:
        # df = pd.DataFrame(dataframe.iloc[0]).transpose()
        # prediction = predict_url(df)

        # if int(prediction) == 0:
        # bool_link_analysis_website = False
        # except Exception:
        # pass

        # number of input elements
        int_input_website = 0

        # find form accompanied by labels with loginwords
        bool_input_login_website = False
        form = soup.find("form")
        try:
            if form:
                inputs = form.find_all("input")

                if inputs:

                    int_input_website = len(inputs)

                    for inp in inputs:
                        try:
                            if inp["type"] == "hidden":
                                hidden_count += 1
                        except Exception:
                            continue

                    label_tags = form.findAll("label")

                    if label_tags:
                        for label_entry in label_tags:
                            if any(str(label_entry.text).__contains__(word) for word in login_list):
                                bool_input_login_website = True

        except Exception:
            pass

        # website has button
        bool_button_website = False
        button_count = get_element_count("button", soup)
        if button_count > 0:
            bool_button_website = True

        # website has meta information
        bool_meta_website = False

        if soup.find("meta"):
            bool_meta_website = True

        # has hidden elements
        bool_hidden_element_website = False
        if hidden_count > 0:
            bool_hidden_element_website = True

        # number of option tags
        int_option_website = get_element_count("option", soup)
        int_option_website = get_element_count("option", soup)

        # number select tags
        int_select_website = get_element_count("select", soup)

        # number th tags
        int_th_website = get_element_count("th", soup)

        # number of tr tags
        int_tr_website = get_element_count("tr", soup)

        # number of table tags
        int_table_website = get_element_count("table", soup)

        # number of href in a tag
        int_href_website = href_count

        # number of list item tags
        int_li_website = get_element_count("li", soup)

        # number of unordered list tags
        int_ul_website = get_element_count("ul", soup)

        # number of ordered list tags
        int_ol_website = get_element_count("ol", soup)

        # number of div tags
        int_div_website = get_element_count("div", soup)

        # number of span tags
        int_span_website = get_element_count("span", soup)

        # number of article tags
        int_article_website = get_element_count("article", soup)

        # number of p tags
        int_p_website = get_element_count("p", soup)

        # number of checkbox tags
        int_checkbox_website = get_element_count("input", soup, "type", "checkbox")

        # number of buttons
        int_button_website = button_count

        # number of images
        int_image_website = get_element_count("img", soup)

        if predict == False:
            entry = FeatureEntryContent(bool_redirect_website=bool_redirect_website,
                                        bool_favicon_website=bool_favicon_website,
                                        bool_content_extern_website=bool_content_extern_website,
                                        int_links_extern_website=int_links_extern_website,
                                        bool_custom_statusbar_website=bool_custom_statusbar_website,
                                        bool_disable_rightclick_website=bool_disable_rightclick_website,
                                        bool_popup_website=bool_popup_website, bool_iframe_website=bool_iframe_website,
                                        bool_action_website=bool_action_website,
                                        bool_action_extern_website=bool_action_extern_website,
                                        bool_form_post_website=bool_form_post_website,
                                        int_phishy_tokens_website=int_phishy_tokens_website,
                                        bool_input_website=bool_input_website,
                                        float_description_sim_website=float_description_sim_website,
                                        bool_bond_status_website=bool_bond_status_website,
                                        bool_freq_domain_extern_website=bool_freq_domain_extern_website,
                                        float_login_home_website=float_login_home_website,
                                        bool_copyright_website=bool_copyright_website,
                                        float_copyright_sim_website=float_copyright_sim_website,
                                        float_title_sim_website=float_title_sim_website,
                                        float_unique_links_website=float_unique_links_website,
                                        # bool_link_analysis_website=bool_link_analysis_website,
                                        int_input_website=int_input_website,
                                        bool_input_login_website=bool_input_login_website,
                                        bool_button_website=bool_button_website,
                                        bool_meta_website=bool_meta_website,
                                        bool_hidden_element_website=bool_hidden_element_website,
                                        int_option_website=int_option_website, int_select_website=int_select_website,
                                        int_th_website=int_th_website,
                                        int_tr_website=int_tr_website, int_table_website=int_table_website,
                                        int_href_website=int_href_website,
                                        int_li_website=int_li_website, int_ul_website=int_ul_website,
                                        int_ol_website=int_ol_website,
                                        int_div_website=int_div_website, int_span_website=int_span_website,
                                        int_article_website=int_article_website,
                                        int_p_website=int_p_website, int_checkbox_website=int_checkbox_website,
                                        int_button_website=int_button_website, int_image_website=int_image_website,
                                        label=label, url=url_orig, final_url=url)

            log(action_logging_enum=INFO,
                logging_text="Processed datapoint. {}".format(url))

            return entry

        elif predict:
            data = {
                "ID": [0],
                "Has Redirect": [bool_redirect_website],
                "Has Favicon": [bool_favicon_website],
                "Has Extern Content": [bool_content_extern_website],
                "Number Extern Links": [int_links_extern_website],
                "Has Custom StatusBar": [bool_custom_statusbar_website],
                "Has Disabled RightClick": [bool_disable_rightclick_website],
                "Has PopUp": [bool_popup_website],
                "Has iFrame": [bool_iframe_website],
                "Has Action": [bool_action_website],
                "Has Extern Action": [bool_action_extern_website],
                "Has Form with POST": [bool_form_post_website],
                "Number PhishyTokens": [int_phishy_tokens_website],
                "Has Input": [bool_input_website],
                "Ratio Description Sim": [float_description_sim_website],
                "Has Bond Status": [bool_bond_status_website],
                "Has Freq Domain Extern": [bool_freq_domain_extern_website],
                "Ratio Similarity": [float_login_home_website],
                "Has Copyright": [bool_copyright_website],
                "Ratio Copyright Sim": [float_copyright_sim_website],
                "Ratio Title Sim": [float_title_sim_website],
                "Ratio Unique Links": [float_unique_links_website],
                "Number Inputs": [int_input_website],
                "Has Input for Login": [bool_input_login_website],
                "Has Button": [bool_button_website],
                "Has Meta": [bool_meta_website],
                "Has Hidden Element": [bool_hidden_element_website],
                "Number Option": [int_option_website],
                "Number Select": [int_select_website],
                "Number TH": [int_th_website],
                "Number TR": [int_tr_website],
                "Number Table": [int_table_website],
                "Number HREF": [int_href_website],
                "Number LI": [int_li_website],
                "Number UL": [int_ul_website],
                "Number OL": [int_ol_website],
                "Number DIV": [int_div_website],
                "Number Span": [int_span_website],
                "Number Article": [int_article_website],
                "Number Paragr": [int_p_website],
                "Number Checkbox": [int_checkbox_website],
                "Number Button": [int_checkbox_website],
                "Number Image": [int_image_website],
                "Label": [label],
                "URL": [url_orig],
                "Final URL": [url]
            }

            columns = list(CONTENT_FEATURE_LIST_COLUMN_NAMES)

            df = pd.DataFrame(data, columns=columns)

            return df

    except Exception as e:
        log(action_logging_enum=WARNING, logging_text=str(e))
        log(action_logging_enum=WARNING, logging_text=str(e.__traceback__))
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        log(ERROR, 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        log(action_logging_enum=WARNING, logging_text="Could not extract content features for {}".format(url))

    log(action_logging_enum=INFO,
        logging_text="Failed datapoint. {}".format(url))
    return None


def extract_features_from_website_list_ray(data):
    """
        extract all features from website list, ray used for parallelism
    """

    @ray.remote
    def get_feature_entry(entry):

        new_entry = extract_features_from_website(entry.url, entry.label, False)

        if not new_entry == None:
            return new_entry

    try:
        result_ids = []
        for entry in data:
            result_ids.append(get_feature_entry.remote(entry))

        feature_list = ray.get(result_ids)

    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")
        feature_list = ray.get(result_ids)

        return feature_list

    return feature_list


def extract_features_from_signature_list(data):
    """
        extract all signature features from list with urls and labels
    """

    @ray.remote
    def get_feature_entry(entry):

        new_entry = extract_features_from_signature(entry.url, entry.label)

        if not new_entry == None:
            return new_entry

    try:
        result_ids = []
        for entry in data:
            result_ids.append(get_feature_entry.remote(entry))

        feature_list = ray.get(result_ids)

    except KeyboardInterrupt as e:
        log(action_logging_enum=ERROR, logging_text="Process interrupted by keyboard signal. Returning the list.")
        feature_list = ray.get(result_ids)

        return feature_list

    return feature_list


def extract_features_from_signature(url, label):
    """
        extract all signature features from url
    """

    common_ents = []
    common_terms = []
    cert_subject = ""

    search_vec = []

    try:
        final_url, redirects, protocol, content = get_redirects(url)

        lang_code = detect_website_language(content)

        # get entities if the language was detected
        if lang_code == "de" or lang_code == "en":
            text = get_website_text(content)

            common_ents = get_common_entities(text, n=5, lang_code=lang_code)
            common_terms = get_common_terms(text=text, n=5, lang_code=lang_code)
            cert_subject = get_ssl_subject(url)

        terms = [[None, 0] for i in range(5)]
        ents = [[None, 0] for i in range(5)]

        for i in range(len(common_ents)):
            ents[i] = common_ents[i]

        for i in range(len(common_terms)):
            terms[i] = common_terms[i]


        if not label == "PREDICT":
            entry = SignatureEntry(url=url, final_url=final_url, label=label, cert_subject=cert_subject, term1=terms[0][0],
                                   term2=terms[1][0], term3=terms[2][0], term4=terms[3][0], term5=terms[4][0],
                                   ent1=ents[0][0], ent2=ents[1][0], ent3=ents[2][0], ent4=ents[3][0], ent5=ents[4][0])

            log(action_logging_enum=INFO, logging_text="Signature extracted from {}.".format(url))
            return entry

        else:
            data = {
                "ID": [0],
                "Term1": [terms[0][0]],
                "Term2": [terms[1][0]],
                "Term3": [terms[2][0]],
                "Term4": [terms[3][0]],
                "Term5": [terms[4][0]],
                "Entity1": [ents[0][0]],
                "Entity2": [ents[1][0]],
                "Entity3": [ents[2][0]],
                "Entity4": [ents[3][0]],
                "Entity5": [ents[4][0]],
                "Label": [label],
                "URL": [url_orig],
                "Final URL": [url]
            }

            columns = list(SIGNATURE_FEATURE_LIST_COLUMN_NAMES)
            entry = pd.DataFrame(data, columns=columns)

            log(action_logging_enum=INFO, logging_text="Signature extracted from {}.".format(url))
            return entry

    except Exception as e:
        log(action_logging_enum=ERROR, logging_text="Could not extract signature from {}. [{}]".format(url, str(e)))
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        log(ERROR, 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

    return None
