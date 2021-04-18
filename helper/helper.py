


# STANDARD LIBARIES

# THIRD PARTY LIBARIES
import bs4
import requests


# LOCAL LIBARIES
from tldextract import extract
from validators import url as validate_url


from config.program_config import INFO, ERROR
from helper.logger import log
from definitions.classes.letter_freqency import FrequencyEntry


######## HELPERS ########
# All helping methods that do not refer to one single module
######## HELPERS ########

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}

# print my IP address
def print_my_IP():
    try:
        ip = requests.get('https://api.ipify.org').text
    except requests.exceptions.RequestException as e:
        log(action_logging_enum=ERROR,
            logging_text="An error occured while querying the IP address.")
        log(action_logging_enum=INFO,
            logging_text="Error description: {err}".format(err=str(e)))

    log(action_logging_enum=INFO, logging_text="My public IP address is: {}".format(ip))


# own score function for error rate
def score_func(y, y_pred):
    counter = 0

    for i in range(len(y)):
        if (y.iloc[i] != y_pred[i]):
            counter += 1

    return counter / len(y)


# get input -> convert to lower case and log system prompt
def get_input_to_lower():
    prompt = input("type a command:")
    prompt = prompt.lower()


    # log system commands
    if prompt != "": log(action_logging_enum=INFO, logging_text="--> " + prompt)

    return prompt


# get letter distribution for German/English
def get_letter_dist(lang_code):

    def german_dist():
        dist_list = []

        dist_list.append(FrequencyEntry("a", 0.0634))
        dist_list.append(FrequencyEntry("b", 0.0221))
        dist_list.append(FrequencyEntry("c", 0.0271))
        dist_list.append(FrequencyEntry("d", 0.0492))
        dist_list.append(FrequencyEntry("e", 0.1599))
        dist_list.append(FrequencyEntry("f", 0.0180))
        dist_list.append(FrequencyEntry("g", 0.0302))
        dist_list.append(FrequencyEntry("h", 0.0411))
        dist_list.append(FrequencyEntry("i", 0.0760))
        dist_list.append(FrequencyEntry("j", 0.0027))
        dist_list.append(FrequencyEntry("k", 0.0150))
        dist_list.append(FrequencyEntry("l", 0.0372))
        dist_list.append(FrequencyEntry("m", 0.0275))
        dist_list.append(FrequencyEntry("n", 0.0959))
        dist_list.append(FrequencyEntry("o", 0.0275))
        dist_list.append(FrequencyEntry("p", 0.0106))
        dist_list.append(FrequencyEntry("q", 0.0004))
        dist_list.append(FrequencyEntry("r", 0.0771))
        dist_list.append(FrequencyEntry("s", 0.0641))
        dist_list.append(FrequencyEntry("t", 0.0643))
        dist_list.append(FrequencyEntry("u", 0.0376))
        dist_list.append(FrequencyEntry("v", 0.0094))
        dist_list.append(FrequencyEntry("w", 0.0140))
        dist_list.append(FrequencyEntry("x", 0.0007))
        dist_list.append(FrequencyEntry("y", 0.0013))
        dist_list.append(FrequencyEntry("z", 0.0122))
        dist_list.append(FrequencyEntry("ä", 0.0054))
        dist_list.append(FrequencyEntry("ö", 0.0024))
        dist_list.append(FrequencyEntry("ü", 0.0063))
        dist_list.append(FrequencyEntry("ß", 0.0015))

        return dist_list

    def english_dist():
        dist_list = []

        dist_list.append(FrequencyEntry("a", 0.084966))
        dist_list.append(FrequencyEntry("b", 0.020720))
        dist_list.append(FrequencyEntry("c", 0.045388))
        dist_list.append(FrequencyEntry("d", 0.033844))
        dist_list.append(FrequencyEntry("e", 0.111607))
        dist_list.append(FrequencyEntry("f", 0.018121))
        dist_list.append(FrequencyEntry("g", 0.024705))
        dist_list.append(FrequencyEntry("h", 0.030034))
        dist_list.append(FrequencyEntry("i", 0.075448))
        dist_list.append(FrequencyEntry("j", 0.001965))
        dist_list.append(FrequencyEntry("k", 0.011016))
        dist_list.append(FrequencyEntry("l", 0.054893))
        dist_list.append(FrequencyEntry("m", 0.030129))
        dist_list.append(FrequencyEntry("n", 0.066544))
        dist_list.append(FrequencyEntry("o", 0.071635))
        dist_list.append(FrequencyEntry("p", 0.031671))
        dist_list.append(FrequencyEntry("q", 0.001962))
        dist_list.append(FrequencyEntry("r", 0.075809))
        dist_list.append(FrequencyEntry("s", 0.057351))
        dist_list.append(FrequencyEntry("t", 0.069509))
        dist_list.append(FrequencyEntry("u", 0.036308))
        dist_list.append(FrequencyEntry("v", 0.010074))
        dist_list.append(FrequencyEntry("w", 0.012899))
        dist_list.append(FrequencyEntry("x", 0.002902))
        dist_list.append(FrequencyEntry("y", 0.017779))
        dist_list.append(FrequencyEntry("z", 0.002722))

        return dist_list

    if lang_code == "en":
        return english_dist()
    else:
        return german_dist()


# get redirects in form of meta tags and 3xx redirects for url
def get_redirects(url, number_of_exec=1):

    # validate url
    if not is_url(url):
        if is_url(str("https://www." + url)):
            url = "https://www." + url

        if is_url(str("https://" + url)):
            url = "https://" + url
        else:
            log(action_logging_enum=ERROR, logging_text="URL does not provide the needed scheme.[{u}]".format(u=url))
            return [-1 , -1, 0, -1]

    # get content for url
    try:
        response = requests.get(url, timeout=3, headers=headers)
    except Exception as e:
        if url.startswith("https://"):
            url = url.replace("https://", "http://", 1)
        elif url.startswith("http://"):
            url = url.replace("http://", "https://", 1)
        try:
            response = requests.get(url, timeout=3, headers=headers)
        except Exception as e:
            log(action_logging_enum=ERROR, logging_text=str(e))
            log(action_logging_enum=ERROR, logging_text="Fatal error while requesting url.[{u}]".format(u=url))
            return [-1, -1, 0, -1]

    protocol = 0
    if response.url.startswith("https://"):
        protocol = 1


    if number_of_exec == 10:
        log(action_logging_enum=ERROR, logging_text="Reached 10 redirects for url.[{u}]".format(u=url))
        return [response.url, 0, protocol, response.content]


    try:
    # find meta redirects
        soup = bs4.BeautifulSoup(response.content.lower(), 'html.parser')
        result = soup.find("meta", attrs={"http-equiv": "refresh"})


        if str(result).lower().__contains__(";") and result:
            wait, text = result["content"].split(";")
            if text.strip().lower().startswith("url="):
                url_follow = text.strip()[4:]
                log(action_logging_enum=INFO, logging_text="Found meta redirect.[{u}]".format(u=url))

                if is_url(url_follow) and url_follow.__ne__(url):
                    exec = number_of_exec + 1
                    url, num, prot, content = get_redirects(url_follow, exec)
                    return [url, num + 1, prot, content]

                if str(url_follow).startswith('/'):
                    if str(url).endswith('/'):
                        url_follow = url + str(url_follow).replace('/', "", 1)
                        exec = number_of_exec + 1
                        url, num, prot, content = get_redirects(url_follow, exec)
                        return [url, num + 1, prot, content]

                if str(response.url).startswith('./'):
                    if str(url).endswith('/'):
                        result = extract(response.url)
                        prot = "http://"
                        if used_prot(response.url) == 1:
                            prot = "https://"
                        url = "{p}{sub}.{dom}.{suf}/{path}".format(p=prot, sub=result.subdomain, dom=result.domain, suf=result.suffix, path=str(response.url).replace('./', '', 1))

                        exec = number_of_exec + 1
                        url, num, prot, content = get_redirects(url_follow, exec)
                        return [url, num + len(response.history), prot, content]
    except Exception as e:
        pass
        # do nothing

    # find 3xx redirects
    if response.history:

        url_follow = response.url

        log(action_logging_enum=INFO, logging_text="Found 3xx redirect.[{u}]".format(u=url))
        if is_url(response.url):
            if response.url.__eq__(url):
                return [url, len(response.history), protocol, response.content]
            exec = number_of_exec + 1
            url, num, prot, content = get_redirects(url_follow, exec)
            return [url, num + len(response.history), prot, content]

        if str(response.url).startswith('/'):
            if str(url).endswith('/'):
                url = url + str(response.url).replace('/', '', 1)
                exec = number_of_exec + 1
                url, num, prot, content = get_redirects(url_follow, exec)
                return [url, num + len(response.history), prot, content]

    return [response.url, 0, protocol, response.content]


# timeout handler for functions without timeout parameter
def handle_timeout(signum, frame):
    log(action_logging_enum=ERROR, logging_text="Timeout based on an exception is called to interrupt.")
    raise Exception("Timeout call")

# detect used protcol 0: http 1: https
def used_prot(url):

    if url.startswith("https://"):
        return 1

    if url.startswith("http://"):
        return 0

    return -1


# check if string is url
def is_url(url):

    # regex could be unprecise
    #regex = re.compile(
    #    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

    if validate_url(url):
        return True

    return False