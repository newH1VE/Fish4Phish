# STANDARD LIBARIES
import itertools
import math
import operator
from threading import Thread
from queue import Queue


# THIRD PARTY LIBARIES
import numpy as np
import pandas as pd
import ray


# LOCAL LIBARIES
from sklearn.feature_selection import mutual_info_classif
from config import configuration as conf
from config.program_config import INFO, ERROR
from helper.logger import log


# LOCAL LIBARIES
def do_greedy_search(data_set, threshold=None, lexical=False, content=False):

    data_set = pd.DataFrame(data_set)
    data_set = data_set.iloc[:500, :]
    y = data_set["Label"]
    x = pd.DataFrame(data_set.drop(["Label"], axis=1))

    number_of_features = len(x.columns)

    conf.add_element("Feature Selection", "Banzhaf Lexical", "")

    banzhaf = get_banzhaf_vector(x, y)
    discrete = []

    for col in x.columns:
        if x[col].dtype == np.bool:
            discrete.append(x.columns.get_loc(col))

    num = 0  # num

    if threshold == None:
        threshold = num

    mi = pd.Series(mutual_info_classif(x, y, discrete_features=discrete))
    mi = mi.values.tolist()

    for i in range(len(mi)):
        mi[i] = [i, mi[i]]

    sumb = 0
    results = []

    for i in range(len(mi)):
        results.append([mi[i][0], (banzhaf[i] * mi[i][1])])

    results = results.sort(key=operator.itemgetter(1), reverse=True)
    log(INFO, "Feature selection completed.")

    for i in range(len(results)):
        print()
        print(x.iloc[:, results[i][0]].name)
        print(results[i][1])

    # save to config
    conf_el = ", ".join(str(e[1]) for e in results)
    conf.set_element("Feature Selection", "NECGT", conf_el)

    return results


def get_banzhaf_vector(x, y):
    banzhaf = []
    # max_coalition = int(math.floor(math.sqrt(len(x.columns))))
    max_coalition = 3
    column_ids = list(range(0, len(x.columns)))
    combinations_list = []
    neighborhood = 0

    log(INFO, "Starting for Banzhaf.")

    def get_feature_contribution_subset(feature_, y_, coalition_, coal_size):

        mutual_information = neighbor_mutual_information_coalition(coalition_, y_, feature_)

        if not mutual_information >= 0:
            log(INFO, "Return contribution.")
            return 0

        log(INFO, "Contribution bigger than 0.")
        # NMI(C_j ; Y | C_i) > NMI(C_j ; Y)
        related = 0
        is_related = False
        for i in range(coal_size):
            nmi_cj_y_ci = neighbor_mutual_information_coalition(coalition_.iloc[:, i], y_, feature_)
            nmi_cj_y = neighbor_mutual_information_coalition(coalition_.iloc[:, i], y_, less=True)

            if nmi_cj_y_ci > nmi_cj_y:
                related += 1

        if related >= (coal_size / 2):
            log(INFO, "Contribution is related.")
            is_related = True


        if not is_related:
            log(INFO, "Return contribution.")
            return 0

        log(INFO, "Return contribution.")
        return 1

    def neighbor_mutual_information_coalition(coalition_, y_, feature_=None, less=False):
        if not less:
            ne_coalition_feature = neighborhood_entropy(coalition_, feature_, union=True)
            ne_feature = neighborhood_entropy(feature_, union=False)

            # union of y and feature column
            y_feature = pd.concat([y_, feature_], axis=1)

            ne_coalition_y_feature = neighborhood_entropy(coalition_, y_feature, union=True)
            ne_y_feature = neighborhood_entropy(y_feature, union=False)

            return (ne_coalition_feature - ne_feature) - (ne_coalition_y_feature - ne_y_feature)

        else:
            ne_cj = neighborhood_entropy(coalition_)
            ne_y = neighborhood_entropy(y_)
            ne_cj_y = neighborhood_entropy(coalition_, y_, union=True)

            return ne_cj + ne_y - ne_cj_y

    def neighborhood_entropy(r, s=None, union=False):

        feature_set = r
        sum_list = []

        if union:
            feature_set = pd.concat([feature_set, s], axis=1)


        n = len(feature_set)

        #@ray.remote
        def do_get(i_):
            g_e = get_neighborhood(feature_set, i_, size_row, distance, feature_vec)
            return math.log((g_e / n), 2)

        try:
            feature_vec = feature_set.values.tolist()
            size_row = len(feature_set)
            distance = 0.15
            result_ids = []
            for i in range(n):
                sum_list.append(do_get(i))


        except Exception as e:
            log(ERROR, str(e))

        sum = 0

        for i in sum_list:
            sum += i

        return (sum * -(1 / n))



    def get_neighborhood(feature_set, i, size_row, distance, feature_vec):
        x_i = feature_vec[i]

        # if single column
        if isinstance(feature_set, pd.Series):
            neigh = 0

            for j in range(size_row):
                x_j = feature_vec[j]
                if isinstance(x_i, bool):
                    res = int(x_i.__ne__(x_j))
                else:
                    res = abs(x_i - x_j)

                if res <= distance:
                    neigh += 1

            neighborhood = neigh

        # if more than ine column / because difference between pd.Series and pd.Dataframe
        else:
            neigh = 0

            for x_j in feature_vec:
                sum_f = 0
                for n in range(len(feature_set.columns)):
                    if isinstance(x_i[n], bool):
                        sum_f += pow(int(x_i[n].__ne__(x_j[n])), 2)
                    else:
                        sum_f += pow(abs(x_i[n] - x_j[n]), 2)

                if math.sqrt(sum_f) <= distance:
                    neigh += 1

            neighborhood = neigh

        return neighborhood

    for i in range(1, max_coalition + 1):
        combinations_list.extend([list(tup) for tup in itertools.combinations(column_ids, i)])



    @ray.remote
    def do_get(i_, y, x, combinations, feature, c, lene):

        res = get_feature_contribution_subset(y_=y, coalition_=x.iloc[:, combinations],
                                                   feature_=feature,
                                                   coal_size=len(combinations))
        log(INFO, "Processed coalition {} of {}.".format(c, lene))
        return res

    ray.init(num_cpus=6)
    len_comb = len(combinations_list)
    for i in range(0, len(column_ids)):

        contrib = 0
        result_ids = []
        for c in range(len_comb):
            combinations = combinations_list[c]
            if i not in combinations:
                feature = x.iloc[:, i]
                result_ids.append(do_get.remote(i, y, x, combinations, feature, c, len_comb))

        contrib_list = ray.get(result_ids)

        for i in contrib_list:
            contrib += i

        res = (1 / len_comb - 1) * contrib
        banzhaf.append(res)

        # save to config
        conf_el = str(conf.get_element("Feature Selection", "Banzhaf Lexical"))
        conf_el += "{}, ".format(str(res))
        conf.set_element("Feature Selection", "Banzhaf Lexical", conf_el)
        log(INFO, "Processed feature {} of {}".format(i, len(column_ids)))

    ray.shutdown()

    return banzhaf
