# STANDARD LIBARIES

# THIRD PARTY LIBARIES
import pandas as pd
import ray

# LOCAL LIBARIES
from definitions.classes.signature_entry import SignatureEntry
from sklearn.base import BaseEstimator, ClassifierMixin
from helper.feature_helper import get_url_components
from components.web_search import bingsearch
from helper.logger import log
from config.program_config import INFO, WARNING

class SignaturClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier for checking signatures of websites
        - given proper nouns/tf/tf-idf token and a domain + TLD
        - it does a web search including all tokens and parses the result
        - if one of the top n results contains the same domain and TLD it returns 1
        - otherwise the classification result is 0
        - naming: Entity1, Entity2, Entity3 ... EntityN; Term1, Term2 ... TermN;
        - naming (cont.) Certificate Subject, URL
    """
    def __init__(self, entities=True, term_freq=True, num_res=10, cert_subject=True, num_ents=2, num_terms=2):
        """
        :param entities: include entities to the web search
        :param term_freq: include tf tokens to the web search
        :param num_res: number of results to analyse for classification
        :param cert_issuer: certificate subject included True/False
        :param num_ents: number of entities to be included in the search (max 5)
        :param num_terms: number of terms to be included in the search (max 5)
        """

        self.entities = entities
        self.term_freq = term_freq
        self.cert_subject = cert_subject
        self.num_res = num_res
        self.num_ents = num_ents
        self.num_terms = num_terms


    def fit(self, x, y):
        """
        Since the classifier does not need to fit for given data, this is a dummy.
        :param x: -
        :param y: -
        :return: -
        """

        return self


    def predict(self, x: pd.DataFrame, y):

        try:
            entities = getattr(self, "entities")
            term_freq = getattr(self, "term_freq")
            num_res = getattr(self, "num_res")
            cert_subject = getattr(self, "cert_subject")
            num_ents = getattr(self, "num_ents")
            num_terms = getattr(self, "num_terms")
        except AttributeError:
            raise RuntimeError("You must initialize the classifier before predicting.")

        ray.init(num_cpus=6)
        ent_index = []
        term_index = []
        cert_index = -1
        url_index = 0

        # get column index for terms, entities, url and cert subject
        for n in range(len(x.columns)):
            name = str(x.iloc[:, n].name)
            if name.lower().__contains__("term") and term_freq and len(term_index) <= num_terms:
                term_index.append(n)

            if name.lower().__contains__("cert") and cert_subject:
                cert_index = n

            if name.lower().__contains__("entit") and entities and len(ent_index) <= num_ents:
                ent_index.append(n)

            if name.lower().__contains__("url"):
                url_index = n

            if len(term_index) > num_terms:
                term_index = term_index[:num_terms]

            if len(ent_index) > num_ents:
                ent_index = ent_index[:num_ents]


        @ray.remote
        def do_predictions(entry, label):
            nonlocal ent_index
            nonlocal entities
            nonlocal term_index
            nonlocal term_freq
            nonlocal cert_subject
            nonlocal url_index
            nonlocal cert_index

            domain = ""
            search = ""
            sub_domain = True

            if isinstance(entry[url_index], str):
                components = get_url_components(entry[url_index])
                if components[2] == "" or components[2] == "www":
                    search = "{}.{}".format(components[3], components[4])
                    sub_domain = False
                else:
                    search = "{}.{}.{}".format(components[2], components[3], components[4])
                domain = search

            if entities and ent_index:
                for j in ent_index:
                    if isinstance(entry[j], str):
                        if len(entry[j]) > 1:
                            search += " {}".format(entry[j])

            if term_freq and term_index:
                for j in term_index:
                    if isinstance(entry[j], str):
                        if len(entry[j]) > 1:
                            search += " {}".format(entry[j])

            if cert_index != -1 and cert_subject and entry[cert_index] is isinstance(entry[cert_index], str):
                search += " {}".format(entry[cert_index])

            if search == "":
                return 1
            else:
                results = bingsearch.search(search, num_res=num_res)

                if results == -1:
                    return 1

                nums = 0

                for res in results:
                    components = get_url_components(res)
                    if not sub_domain:
                        res_domain = "{}.{}".format(components[3], components[4])
                    else:
                        res_domain = "{}.{}.{}".format(components[2], components[3], components[4])

                    if domain == res_domain:
                        log(INFO, "Processed datapoint.")
                        if label == 1:
                            log(WARNING, "FN: {}".format(search))
                        nums += 1

                if nums > 0:
                    return 0

                return 1

                nums = 0
                search = domain

                results = bingsearch.search(domain, num_res=num_res)

                if results == -1 or len(results) == 0:
                    return 1

                for res in results:
                    components = get_url_components(res)
                    if not sub_domain:
                        res_domain = "{}.{}".format(components[3], components[4])
                    else:
                        res_domain = "{}.{}.{}".format(components[2], components[3], components[4])

                    if domain == res_domain:
                        log(INFO, "Processed datapoint.")
                        if label == 1:
                            log(WARNING, "FN: {}".format(search))
                        nums += 1

                if nums > 0:
                    return 0

                if label == 0:
                    log(WARNING, "FP: {}".format(search))



                log(INFO, "Processed datapoint.")
                return 1


        x_list = x.values.tolist()
        result_ids = []
        for i in range(len(x_list)):
            result_ids.append(do_predictions.remote(x_list[i], y[i]))

        predictions = ray.get(result_ids)
        ray.shutdown()

        return predictions





