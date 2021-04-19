

# STANDARD LIBARIES
import time

# THIRD PARTY LIBARIES
import requests

# LOCAL LIBARIES
from config.program_config import WARNING, INFO, ERROR, BING_SEARCH_KEY
from helper.logger import log


# search terms in bing, num_res defines maximum nunber of results de be returned
def search(term, wait_after_429=True, num_res=10):

    def fetch_results(headers, params):

        search_url = "https://api.bing.microsoft.com/v7.0/search"

        try:
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()

        except Exception as e:
            try:
                time.sleep(2)
                response = requests.get(search_url, headers=headers, params=params)
                response.raise_for_status()
            except:
                log(action_logging_enum=ERROR, logging_text="[Function search]: An error occured while querying the Bing API.")
                log(action_logging_enum=INFO,
                logging_text="[Function search]: Error description: {err}".format(err=str(e)))
                return -1

        status_code = int(response.status_code)

        if status_code >=200 and status_code < 400:
            return response.json()
        else:
            if wait_after_429:
                while status_code < 200 or status_code >= 400:
                    log(action_logging_enum=WARNING, logging_text="Bing Search returned Status Code 429. "
                                                            "Next check in: 5 minutes.")
                    time.sleep(300)
                    try:
                        response = requests.get(search_url, headers=headers, params=params)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        log(action_logging_enum=ERROR,
                            logging_text="An error occured while querying the Bing API.")
                        log(action_logging_enum=INFO,
                            logging_text="Error description: {err}".format(err=str(e)))

                return response.json()

        return -1

    def parse_results(search_results, num_res):
        search_results_list = []
        try:
            if search_results["webPages"]:
                for i in search_results["webPages"]["value"]:
                    search_results_list.append(i["url"])

            if num_res < len(search_results_list):
                search_results_list = search_results_list[:num_res]
                return search_results_list

            return search_results_list
        except Exception:
            print(search_results)
            return search_results_list


    subscription_key = BING_SEARCH_KEY
    assert subscription_key
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": term, "textDecorations": True, "textFormat": "HTML"}

    search_results = fetch_results(headers=headers, params=params)

    if search_results == -1:
        return -1

    return list(parse_results(search_results, num_res))

