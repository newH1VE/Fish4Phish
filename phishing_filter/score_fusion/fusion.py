# STANDARD LIBARIES


# THIRD PARTY LIBARIES
import ray


# LOCAL LIBARIES
from config.program_config import DATA_PATH, INFO, WARNING, ERROR
from helper.logger import log



def majority(score_1, score_2, score_3):

    def majority_int(_score_1, _score_2, _score_3):
        result = _score_1 + _score_2 + _score_3

        if result > 1:
            return 1
        else:
            return 0

    @ray.remote
    def ray_majority_int(_score_1, _score_2, _score_3):
        return majority_int(_score_1, _score_2, _score_3)

    def majority_list(_score_list_1, _score_list_2, _score_list_3):
        if not len(_score_list_1).__eq__(len(_score_list_2)) or not len(_score_list_1).__eq__(len(_score_list_3)):
            log(ERROR, "Lists have not the same length.")
            return

        ray.init(num_cpus=6)
        result_ids = []

        for i in range(len(_score_list_1)):
            _score_1 = _score_list_1[i]
            _score_2 = _score_list_2[i]
            _score_3 = _score_list_3[i]
            result_ids.append(ray_majority_int.remote(_score_1, _score_2, _score_3))

        prediction_list = ray.get(result_ids)
        ray.shutdown()
        return prediction_list


    if isinstance(score_1, int) and isinstance(score_2, int) and isinstance(score_3, int):
        return majority_int(score_1, score_2, score_3)
    else:
        return majority_list(score_1, score_2, score_3)


def majority_weight(score_1, score_2, score_3, threshold=0.3):
    """
    :param score_1: score of lexical filter
    :param score_2: score of content filter
    :param score_3: score of signature filter
    :param threshold: threshold for score >= threshold final classification is phishing
    :return: weighted score for final classification
    """


    def majority_weight(_score_1, _score_2, _score_3, threshold):
        weight_lex = 0.9152
        weight_con = 0.8607
        weight_sign = 0.8528


        result = ((_score_1[1]*weight_lex) + (_score_2[1]*weight_con) + (_score_3*weight_sign))/3

        if result >= threshold:
            return 1
        else:
            return 0

    @ray.remote
    def ray_majority_weight(_score_1, _score_2, _score_3, threshold):
        return majority_weight(_score_1, _score_2, _score_3, threshold)

    def majority_list(_score_list_1, _score_list_2, _score_list_3, threshold):
        if not len(_score_list_1).__eq__(len(_score_list_2)) or not len(_score_list_1).__eq__(len(_score_list_3)):
            log(ERROR, "Lists have not the same length.")
            return

        #ray.init(num_cpus=6)
        result_ids = []

        for i in range(len(_score_list_1)):
            _score_1 = _score_list_1[i]
            _score_2 = _score_list_2[i]
            _score_3 = _score_list_3[i]
            result_ids.append(ray_majority_weight.remote(_score_1, _score_2, _score_3, threshold))

        prediction_list = ray.get(result_ids)
        #ray.shutdown()
        return prediction_list

    if isinstance(score_1, int) and isinstance(score_2, int) and isinstance(score_3, int):
        return majority_weight(score_1, score_2, score_3, threshold)
    else:
        return majority_list(score_1, score_2, score_3, threshold)


def get_f1(pred, labels):

    if len(pred) != len(labels):
        return None

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(labels)):
        if labels[i] == 1:
            if pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred[i] == 1:
                fp += 1
            else:
                tn += 1

    p = tp/(tp+fp)
    r = tp / (tp + fn)
    f1 = (2*p*r)/(p+r)

    log(INFO, "Precision: {}".format(str(p)))
    log(INFO, "Recall: {}".format(str(r)))
    log(INFO, "F1: {}".format(str(f1)))

    return f1


