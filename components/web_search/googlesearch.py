

# STANDARD LIBARIES
import time

# THIRD PARTY LIBARIES
from bs4 import BeautifulSoup
import requests

# LOCAL LIBARIES
from config.program_config import WARNING, INFO, ERROR
from helper.logger import log


# search for term in google
def search(term, num_results=10, lang="en", wait_after_429=True):
    usr_agent = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/61.0.3163.100 Safari/537.36'}

    def fetch_results(search_term, number_results, language_code):
        escaped_search_term = search_term.replace(' ', '+')

        google_url = 'https://www.google.de/search?q={}&num={}&hl={}'.format(escaped_search_term, number_results + 1, language_code)

        try:
            response = requests.get(google_url, headers=usr_agent, timeout=10)
            response.raise_for_status()
        except Exception as e:
            try:
                time.sleep(2)
                response = requests.get(google_url, headers=usr_agent, timeout=10)
            except Exception as e:
                log(action_logging_enum=ERROR, logging_text="[Function search]: An error occured while querying the Google API.")
                log(action_logging_enum=INFO,
                logging_text="[Function search]: Error description: {err}".format(err=str(e)))
                return -1

        status_code = int(response.status_code)


        if status_code >=200 and status_code < 400:
            return response.content
        else:
            if wait_after_429:
                while status_code < 200 or status_code >= 400:
                    log(action_logging_enum=WARNING, logging_text="Google Search returned Status Code 429. "
                                                            "Next check in: 5 minutes.")
                    time.sleep(300)
                    try:
                        response = requests.get(google_url, headers=usr_agent, timeout=10)
                        status_code = int(response.status_code)
                    except Exception as e:
                        log(action_logging_enum=ERROR,
                            logging_text="An error occured while querying the Google API.")
                        log(action_logging_enum=INFO,
                            logging_text="Error description: {err}".format(err=str(e)))
                return response.content

        return -1

    def parse_results(raw_html):

        soup = BeautifulSoup(raw_html, 'html.parser')
        result_block = soup.find_all('div', attrs={'class': 'g'})
        for result in result_block:
            link = result.find('a', href=True)
            title = result.find('h3')
            if link and title:
                yield link['href']

    html = fetch_results(term, num_results, lang)
    return list(parse_results(html))

