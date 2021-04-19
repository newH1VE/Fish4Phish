# STANDARD LIBARIES
import math
import re
import ssl
from datetime import datetime
import string
from difflib import SequenceMatcher
from queue import Queue
from threading import Thread
from string import ascii_lowercase
from operator import attrgetter
import sys
import linecache
import socket
from collections import Counter

# THIRD PARTY LIBARIES
import ray
from langdetect import detect
import spacy
import tldextract
import bs4
import requests
from validators import url as validate_url
from selenium import webdriver
from tldextract import extract
import OpenSSL
import whois
from urllib.parse import urljoin, urlparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from translate import Translator
from cryptography.x509.oid import NameOID

# LOCAL LIBARIES
from config.program_config import INFO, ERROR, WARNING
from helper.logger import log
from components.web_search import googlesearch, bingsearch
from helper.helper import get_redirects, is_url, get_letter_dist
from definitions.classes.redirect_entry import RedirectEntry
from definitions.classes.letter_freqency import FrequencyEntry

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
nlp_de = spacy.load("de_core_news_md")
nlp_en = spacy.load("en_core_web_sm")
translator_en = Translator(to_lang='en', from_lang='de')


"""
All helping methods for feature extraction
"""



# return true if host contains ip, false if not
def is_IP(host):
    ip_matcher = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    if ip_matcher.match(host):
        return True

    return False

# compute kullback-leibler divergence of letter distribution of links
def compute_divergence(url, content):

    def preprocessing(s):
        r_string = ""

        for char in s:
            if str(char).isalpha():
              r_string += char

        return r_string.lower()

    def get_freqency(s, speech_dist):
        frequency = []

        for item in speech_dist:
            char = item.letter
            frequency.append(FrequencyEntry(str(char), float(str(s).count(char)/len(s))))

        return frequency

    def divergence(url_dist, speech_dist):
        url_dist.sort(key=attrgetter('letter'))
        speech_dist.sort(key=attrgetter('letter'))
        sum_div = 0

        for index,item in enumerate(url_dist):
            if float(item.frequency) > 0:
                log = math.log((float(item.frequency)/float(speech_dist[index].frequency)), 2)
                sum_div += float(float(item.frequency)*log)

        return sum_div

    lang_code = detect_website_language(content.lower())

    if lang_code == "en" or lang_code == "de":
        speech_dist = get_letter_dist(lang_code)
        url_dist = get_freqency(preprocessing(url), speech_dist)

        return divergence(url_dist, speech_dist)

    return -1


# get last_updated for de domain and creation date for other domains
def get_whois(url):
    extracted_url = tldextract.extract(url)

    domain = extracted_url.domain
    suffix = extracted_url.suffix

    host = "{}.{}".format(domain, suffix)

    try:
        whois_host = whois.query(host)
        if suffix == "de":
            last_updated = whois_host.last_updated
            return last_updated, None

        expiry_date = whois_host.expiration_date
        creation_date = whois_host.creation_date
        return creation_date, expiry_date

    except Exception as e:
        log(action_logging_enum=ERROR, logging_text=str(e))
        log(action_logging_enum=ERROR,
            logging_text="Could not get creation/last updated date for host: {}".format(host))

    return None, None


# get expiration date of domain by whois
def get_whois_expiry_date(url):
    extracted_url = tldextract.extract(url)

    domain = extracted_url.domain
    suffix = extracted_url.suffix
    host = "{}.{}".format(domain, suffix)

    try:
        whois_host = whois.query(host)
        expiry_date = whois_host.expiration_date
    except Exception as e:
        # log(action_logging_enum=ERROR, logging_text=str(e))
        log(action_logging_enum=ERROR, logging_text="Could not find expiry date by whois query for: {}".format(host))
        return None

    return expiry_date


# get ssl expiration date and creation date
def get_ssl_information(url):
    try:
        extracted_url = tldextract.extract(url)
        domain = extracted_url.domain
        suffix = extracted_url.suffix

        host = str("{}.{}".format(domain, suffix))
        ip_host = socket.gethostbyname(host)

        cert = ssl.get_server_certificate((ip_host, 443))
        x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
        expire_string = x509.get_notAfter().decode("utf-8")[:8]
        creation_string = x509.get_notBefore().decode("utf-8")[:8]
        expire_date = datetime.strptime(expire_string, '%Y%m%d')
        creation_date = datetime.strptime(creation_string, '%Y%m%d')
        return expire_date, creation_date
    except Exception as e:
        # log(action_logging_enum=ERROR, logging_text=str(e))
        log(action_logging_enum=ERROR, logging_text="Could not get certificate/handshake failure for: {}".format(host))

        return None, None


# get ssl expiration date and creation date
def get_ssl_subject(url):
    try:
        extracted_url = tldextract.extract(url)
        domain = extracted_url.domain
        suffix = extracted_url.suffix

        host = str("{}.{}".format(domain, suffix))
        ip_host = socket.gethostbyname(host)

        cert = ssl.get_server_certificate((ip_host, 443))
        x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
        return x509.get_subject().O


    except Exception as e:
        log(action_logging_enum=ERROR, logging_text="Could not get certificate/handshake failure for: {}".format(host))
        return None
    return None


def find_extern_links(content, domain, suffix, url):
    found = False
    number_extern_links = 0

    try:
        soup = bs4.BeautifulSoup(content.lower(), 'lxml')

    except Exception as e:
        log(action_logging_enum=ERROR, logging_text="Couldn't load content of website for external content check.")

    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue

        try:
            if not validate_url(href):
                if validate_url(urljoin(url, href)):
                    href = urljoin(url, href)
                else:
                    continue

            components = get_url_components(href)

            extracted_domain = components[3]
            extracted_suffix = components[4]

            if extracted_domain.__ne__(domain) or extracted_suffix.__ne__(suffix):
                number_extern_links += 1
                found = True

        except Exception as e:
            continue

    return found, number_extern_links


# get redirects by multithreading
def get_redirects_list(data):
    def do_ray():

        redirect_object_list = []

        #ray.init(num_cpus=6)

        @ray.remote
        def redirects(href):
            href_url, href_num_redirects, href_protocol, href_content = get_redirects(href)

            if not href_content == -1:
                redirect_object = RedirectEntry(url=href_url, redirects=href_num_redirects, protocol=href_protocol,
                                                content=href_content)
                return redirect_object

        result_ids = []
        for href in data:
            result_ids.append(redirects.remote(href))

        redirect_object_list = ray.get(result_ids)

        return redirect_object_list

    def do_threading():

        threads = 13
        redirect_object_list = []

        def get_feature_entry():
            while True:
                href = q.get()
                href_url, href_num_redirects, href_protocol, href_content = get_redirects(href)

                if not href_content == -1:
                    redirect_object = RedirectEntry(url=href_url, redirects=href_num_redirects, protocol=href_protocol,
                                                    content=href_content)
                    append_in_list(redirect_object)

                q.task_done()

            return

        def append_in_list(new_entry):
            nonlocal redirect_object_list

            if not new_entry == None:
                redirect_object_list.append(new_entry)

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
            return redirect_object_list

        return redirect_object_list

    return do_threading()
    #return do_ray()



# get website text
def get_website_text(content):
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, bs4.element.Comment):
            return False
        return True

    def clean_strings(texts):
        cleaned_texts = []
        try:

            for string in texts:
                string = str(string)
                contains = True

                while contains:
                    if string.__contains__('\\'):
                        pos = string.find('\\')
                        replace_seq = ""

                        for i in range(pos, len(string)):
                            if not string[i] == " ":
                                replace_seq += string[i]
                            else:
                                break

                        # replace sequence
                        string = string.replace(replace_seq, "")
                        # remove double whitespace
                        if len(string) > 0 and string.__contains__(' '):
                            string = str(string).replace("  ", " ")

                    else:
                        contains = False
                        break
                if len(string) > 0:
                    cleaned_texts.append(string.strip())

        except Exception:
            return texts

        return cleaned_texts


    soup = bs4.BeautifulSoup(content.decode("utf-8"), 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    visible_texts = clean_strings(visible_texts)

    return u" ".join(t.strip() for t in visible_texts)

    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False

    if isinstance(element, bs4.element.Comment):
        return False
    return True


# get fqdn, scheme, subdomain, domain, suffix, port, path, query, fragment
def get_url_components(url):
    # backup regex
    # ip_regex = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

    # preprocessing > check for @ in fqdn
    url_pre_processing = urlparse(url)
    if url_pre_processing.netloc.__contains__("@"):
        at_count = url_pre_processing.netloc.count("@")

        url = url_pre_processing.scheme + "://" + url.split("@", at_count)[at_count]

    url_extracted = tldextract.extract(url)
    url_parsed = urlparse(url)

    fqdn = url_extracted.fqdn
    scheme = url_parsed.scheme
    subdomain = url_extracted.subdomain
    domain = url_extracted.domain
    suffix = url_extracted.suffix
    port = ""
    path = ""
    if url_parsed.path:
        path = url_parsed.path

    netloc = url_parsed.netloc

    if netloc.__contains__("@"):
        netloc = netloc.split("@")[1]

    query = ""
    if url_parsed.query:
        query = "?" + url_parsed.query

    fragment = url_parsed.fragment

    if fragment:
        fragment = "#" + fragment

    res = re.findall(r'[0-9]+(?:\.[0-9]+){3}', netloc)

    if res:
        domain = res[0]
        fqdn = scheme + "://" + netloc
        subdomain = netloc.split(domain)[0]
        last_occurence = subdomain.rsplit(".", 1)
        subdomain = "".join(last_occurence)

    if netloc.__contains__(":"):
        port = netloc.split(":")[1]

    return [fqdn, scheme, subdomain, domain, suffix, port, path, query, fragment]


# perform google query to get the login page
def search_login_page(url, login_token_list, selenium_analysis=False):
    def do_selenium_search(driver, content):
        # search by driver for clickable links
        try:
            driver.get(url)

        except Exception as e:
            log(action_logging_enum=INFO, logging_text=str(e))
            return url, False

        for token in login_token_list:
            if token not in str(content.lower()):
                continue

            try:
                # find button/a Anchor that contains login token (case-insensitive)
                xpath = "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'),'{f}')]".format(
                    f=token)
                button = driver.find_element_by_xpath(xpath)
                button.click()

            except Exception as e:
                log(action_logging_enum=INFO, logging_text=str(e))
                continue

            login_url = str(driver.current_url)
            log(action_logging_enum=INFO,
                logging_text="Login page found by selenium analysis.")
            return login_url, True

        return url, False

    def do_content_search(content):
        soup = bs4.BeautifulSoup(content.lower(), 'html.parser')
        if soup.find('a', href=True):
            for link in soup.find_all('a', href=True):
                if any(link.text.__eq__(token.lower()) for token in login_token_list):
                    login_url = str(link['href']).strip()

                    if login_url.startswith('/') and url.endswith('/'):
                        login_url = url + login_url.replace('/', '', 1)

                    if login_url.startswith('./'):
                        if url.endswith('/'):
                            login_url = url + login_url.replace('./', '', 1)
                        else:
                            login_url = url + login_url.replace('.', '', 1)

                    if validate_url(login_url):
                        log(action_logging_enum=INFO,
                            logging_text="Login page found by content analysis.")
                        return login_url, True

        return url, False

    def do_bing_search(domain, ccTLD):
        return (bingsearch.search("{d}.{s} login".format(d=domain, s=ccTLD), wait_after_429=False))

    def do_google_search(domain, ccTLD):
        return (googlesearch.search("{d}.{s} login".format(d=domain, s=ccTLD), 5, "de", wait_after_429=False))

    def do_search(domain, ccTLD, url):

        try:
            result = do_bing_search(domain, ccTLD)
        except Exception as e:
            result = -1
            log(action_logging_enum=WARNING, logging_text=str(e))

        if result == -1:
            try:
                result = do_google_search(domain, ccTLD)
            except Exception as e:
                result = -1
                log(action_logging_enum=ERROR, logging_text=str(e))

            if result == -1:
                return url, False

        for entry in result:
            orig_url = url
            extracted_search = extract(entry)
            search_domain = extracted_search.domain
            search_cctld = extracted_search.suffix
            found_url = str(entry)

            if cut_protocol(orig_url).__ne__(cut_protocol(found_url)):
                orig_url = orig_url + "/"

            if str(search_domain).__eq__(domain) and not cut_protocol(found_url).__eq__(cut_protocol(orig_url)):

                if str(ccTLD).__ne__(str(search_cctld)) and cut_protocol(found_url).__contains__("/"):
                    path = cut_protocol(found_url).split("/", 1)[1]

                    if not path:
                        continue

                if entry.startswith("http://") and orig_url.startswith("https://"):
                    found_url = found_url.replace("http://", "https://", 1)
                    try:
                        requests.get(found_url, timeout=10, headers=headers)
                    except Exception as e:
                        found_url = found_url.replace("https://", "http://", 1)

                login_url = found_url
                log(action_logging_enum=INFO,
                    logging_text="Login page found by search engine.")
                return login_url, True

        return url, False

    if selenium_analysis:
        driver = webdriver.Chrome()
        driver.implicitly_wait(20)

    if is_url(str("https://www." + url)):
        url = "https://www." + url

    if is_url(str("https://" + url)):
        url = "https://" + url

    if not url.startswith("https://") and not url.startswith("http://"):
        url = "https://" + url

    if not is_url(url):
        log(action_logging_enum=ERROR, logging_text="URL does not provide the needed scheme. [{}]".format(url))
        return url, False

    extracted = extract(url)
    domain = extracted.domain
    ccTLD = extracted.suffix

    login_url, changed_status = do_search(domain, ccTLD, url)

    # get redirects for url
    try:
        url, num_redirects, protocol, content = get_redirects(url)
    except Exception as e:
        log(action_logging_enum=ERROR, logging_text="Error while getting redirects for url: {}".format(url))
        return None, False

    if protocol == -1 and content == -1:
        return None, False

    lang_website = detect_website_language(content)

    if lang_website == None:
        log(action_logging_enum=WARNING, logging_text="Website does not use German or English.[{}]".format(url))
        return None, False

    if not changed_status:

        if changed_status.__eq__(False):
            login_url, changed_status = do_content_search(content)

        if changed_status.__eq__(False) and selenium_analysis:
            login_url, changed_status = do_selenium_search(driver, content)

        if selenium_analysis:
            driver.close()

        if changed_status.__eq__(False):
            log(action_logging_enum=WARNING, logging_text="No login page found for: " + url)
            return url, False

    log(action_logging_enum=INFO, logging_text="Login page found: " + login_url)

    return login_url, changed_status


def cut_protocol(url):
    if url.startswith("https://"):
        url = url.replace("https://", "", 1)

    if url.startswith("http://"):
        url = url.replace("http://", "", 1)

    return url


# compute entropy for string
def entropy_of_string(s):
    full_list = list(s)
    unique_list = list(set(full_list))

    val = 0
    size = len(full_list)

    for char in unique_list:
        p_i = full_list.count(char) / size

        val = val + p_i * math.log(p_i, 2)

    val = -val

    return val


# remove http:// and https:// from urls in list
def remove_chars_from_URL(url):
    if url.startswith("http://"):
        url = url.replace('http://', '')

    if url.startswith("https://"):
        url = url.replace('https://', '')

    if url.startswith("www."):
        url = url.replace('www.', '')

    return url


# detect used protcol 0: http 1: https
def request_used_protocol(url):
    if not url.lower().startswith("https://") and not url.lower().startswith("http://"):
        url = "https://" + url

    try:
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        if url.startswith("https://"):
            return True
        else:
            return False
    except Exception as e:
        log(action_logging_enum=ERROR, logging_text=str(e))

    return False


# reformat labels from string to int (phish:1, benign:0)
def binarize_labels(list_obj):
    for entry in list_obj:
        if entry.label == "Phish":
            entry.label = 1
        if entry.label == "Benign":
            entry.label = 0
    return list_obj


# remove http:// and https:// from urls in list
def remove_chars_from_list(listobj):
    for i in range(len(listobj)):
        urlstring = str(listobj[i].url)
        if urlstring.startswith("http://"):
            urlstring = urlstring.replace('http://', '')
        if urlstring.startswith("https://"):
            urlstring = urlstring.replace('https://', '')
        if urlstring.startswith("www."):
            urlstring = urlstring.replace('www.', '')
        listobj[i].url = urlstring
    return listobj



def get_common_entities(text,n=5, lang_code="de"):
    ents = get_entities(text, lang_code)

    if not ents == None:
        if len(ents) >= n:
            return ents[:n]
        else:
            return ents[:len(ents)]

    return None


# extract named entites from texts
def get_entities(text, lang_code="de"):

    # get named entities from english texts
    def get_named_entities_en(text):
        doc = nlp_en(text)
        ents = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
        ents_freq = []

        for ent in ents:
            ents_freq.append([str(ent).strip(), text.count(ent)])

        ents_freq.sort(key=lambda x: x[1], reverse=True)

        return ents_freq

    # extract named entities from german texts
    def get_named_entities_de(text):
        doc = nlp_de(text)
        ents = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
        ents_freq = []

        for ent in ents:
            ents_freq.append([str(ent).strip(), text.count(ent)])

        ents_freq.sort(key=lambda x: x[1], reverse=True)

        return ents_freq

    if lang_code == "de":
        return get_named_entities_de(text)
    elif lang_code == "en":
        return get_named_entities_en(text)

    return None



# do term frequency on text get 5 most used terms
def get_common_terms(text, n=5, lang_code="de"):

    def get_common_terms_en(text):
        doc = nlp_en(text)
        words = [token.text for token in doc if (not token.is_stop and not token.is_punct and (token.pos_ == "NOUN" or token.pos_ == "PROPN"))]
        word_freq = []

        for word in list(set(words)):
            word_freq.append([str(word).strip(), text.count(word)/len(words)])

        word_freq.sort(key=lambda x: x[1], reverse=True)
        return word_freq

    def get_common_terms_de(text):
        doc = nlp_de(text)
        words = [token.text for token in doc if (not token.is_stop and not token.is_punct and (token.pos_ == "NOUN" or token.pos_ == "PROPN"))]
        word_freq = []

        for word in words:
            word_freq.append([str(word).strip(), text.count(word)/len(words)])

        word_freq.sort(key=lambda x: x[1], reverse=True)
        return word_freq

    def get_common_terms_nolang(text):
        words = text.split()
        word_freq = []

        for word in words:
            word_freq.append([str(word).strip(), text.count(word)/len(words)])

        word_freq.sort(key=lambda x: x[1], reverse=True)
        return word_freq

    terms = None

    if lang_code == "de":
        terms = get_common_terms_de(text)
    elif lang_code == "en":
        terms = get_common_terms_en(text)
    elif not lang_code == "de" and not lang_code == "en":
        terms = get_common_terms_nolang(text)

    if terms == None:
        return None
    else:
        if len(terms) >= n:
            return terms[:n]
        else:
            return terms[:len(terms)]


# search for results from keywords
def do_search(domain, suffix, url, words):
    search_string = ""

    search_string = "{d}.{s}".format(d=domain, s=suffix)

    for word in words:
        if not word == None:
            search_string += " {w}".format(w=word)

    try:
        result = bingsearch.search(search_string, wait_after_429=False)
    except Exception as e:
        result = -1
        log(action_logging_enum=WARNING, logging_text=str(e))

    if result == -1:
        return None

    for entry in result:
        extracted_search = get_url_components(str(entry))
        search_domain = extracted_search[3]
        search_cctld = extracted_search[4]
        found_url = str(entry)

        if str(search_domain).__eq__(domain) and str(suffix).__eq__(str(search_cctld)) :

            log(action_logging_enum=INFO,
                logging_text="Login page found by search engine.")
            return found_url, True

    return url, False


# detect language used by website
def detect_website_language(content):
    lang = ""
    try:
        soup = bs4.BeautifulSoup(content.decode("utf-8").lower(), "html.parser")
        html_tag = soup.find('html')
        lang = html_tag["lang"]
        log(action_logging_enum=INFO, logging_text="Detected language by html tag: {}".format(lang))
    except Exception:
        try:
            soup = bs4.BeautifulSoup(content.decode("utf-8"), "lxml")
            html_tag = soup.find('html')
            lang = html_tag["lang"]
            log(action_logging_enum=INFO, logging_text="Detected language by html tag: {}".format(lang))
        except Exception:
            pass

    if lang.lower().__contains__("de"):
        return "de"

    if lang.lower().__contains__("en"):
        return "en"

    try:
        meta = soup.find('meta', attrs={'http-equiv': 'content-language'})
        lang = meta['content']
        log(action_logging_enum=INFO, logging_text="Detected language by title tag: {}".format(lang))
        if lang.lower().__contains__("de"):
            return "de"
        elif lang.lower().__contains__("en"):
            return "en"
        else:
            return None
    except Exception as e:
        pass

    try:
        soup = bs4.BeautifulSoup(content.decode("utf-8"), "html.parser")
        title = soup.find('title').text
        lang = detect_language(title)
        log(action_logging_enum=INFO, logging_text="Detected language by title tag: {}".format(lang))
        return lang

    except Exception as e:
        log(action_logging_enum=ERROR, logging_text=str(e))
        log(action_logging_enum=ERROR,
            logging_text="Could not detect language of website. Could not find website title.")

    return None


# create tokens of test
def word_tokenize(text, all_stopwords=None, lang_code=-1):

    if not lang_code == -1:
        if lang_code == "de":
            all_stopwords = nlp_de.Defaults.stop_words

        if lang_code == "en":
            all_stopwords = nlp_en.Defaults.stop_words

    if all_stopwords == None:
        return text

    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if not word in all_stopwords])

    return text



# calculare string similarity by jaccard similarity
def string_similarity(s1, s2):

    def get_cosine_similarity(s1, s2):
        try:
            lang_s1 = detect_language(s1)
            lang_s2 = detect_language(s2)

            nlp = None

            if lang_s1 == lang_s2 and not lang_s1 == None:
                if lang_s1 == "de":
                    nlp = nlp_de
                else:
                    nlp = nlp_en
            elif not lang_s2 == lang_s1:
                log(action_logging_enum=INFO, logging_text="Strings have different languages. Translate ...")
                if lang_s2 == 'de':
                    s2 = translator_en.translate(s2)
                else:
                    s1 = translator_en.translate(s1)
                nlp = nlp_en
            else:
                log(action_logging_enum=INFO, logging_text="Language was not detected. Using Python SequenceMatcher.")
                return SequenceMatcher(None, s1, s2).ratio()

            all_stopwords = nlp.Defaults.stop_words

            # tokens_s1 = word_tokenize(s1, all_stopwords)
            # tokens_s2 = word_tokenize(s2, all_stopwords)

            texts = [s1, s2]

            tokens = list(map(word_tokenize, texts, all_stopwords))

            vectorizer = CountVectorizer().fit_transform(raw_documents=tokens)
            vectors = vectorizer.toarray()

            vec_s1 = vectors[0].reshape(1, -1)
            vec_s2 = vectors[1].reshape(1, -1)

            return round(cosine_similarity(vec_s1, vec_s2)[0][0], 4)

        except Exception as e:
            log(action_logging_enum=ERROR, logging_text=str(e))
            log(action_logging_enum=INFO, logging_text="Using Python SequenceMatcher.")

        return SequenceMatcher(None, s1, s2).ratio()

    def get_jaccard_index(s1, s2):
        a = set(s1.split())
        b = set(s2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    return get_jaccard_index(s1, s2)


def get_element_count(element, soup, option="", value=""):
    elements = soup.findAll(element)
    option_count = 0

    if not option or not value:
        return len(elements)
    else:
        for el in elements:
            try:
                if value:
                    if el[option] == value:
                        option_count += 1
                else:
                    el[option]
                    option_count += 1
            except Exception:
                continue

    return option_count


# detect language from string
def detect_language(text):
    lang = detect(text)

    if lang == 'de':
        return lang

    if lang == 'en':
        return lang

    return None
