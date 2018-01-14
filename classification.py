# encoding utf-8


from distance import *
import numpy as np
import random
import os.path
import re


def get_data_set(data_set_path="output/"):
    """
    Return set of files corresponding to the data set.

    :param data_set_path: [str] directory name in which .npy files are
    :return: [Dictionary] set of files
    """
    files = os.listdir(data_set_path)
    data_set = {}

    regex = r'(\d+)_(\w+)_(\d+)\.npy'
    for file in files:
        r = re.search(regex, file)

        try:
            number = int(r.group(1))
            person = r.group(2)
            index = int(r.group(3))
        except IndexError:
            print(file, " not well formatted")
        else:
            if person != "jason":
                data_set[number] = data_set[number] + [file] if number in data_set.keys() else [file]

    return data_set


def k_nearest_neighbours(k, test_data_set, ref_data_set, data_set_path="output/", wd=1.0, wh=1.0, wv=1.0):
    """
    Apply k nearest neighbours with a set of reference data and a set of test data.

    :param k: [int] number of neighbours to consider
    :param test_data_set: [list(str)] list of file names for each test data
    :param ref_data_set: [list(str)] list of file names for each reference data
    :param data_set_path: [str] directory name in which .npy files are
    :param wd: [float] value for wd coefficient
    :param wv: [float] value for wv coefficient
    :param wh: [float] value for wh coefficient
    :return: [list of bool] list of booleans. True if a test data is correctly classified.
    """
    regex = r'(\d+)_(\w+)_(\d+)\.npy'

    mat_test_data_set = [np.load(data_set_path + file) for file in test_data_set]
    mat_ref_data_set = [np.load(data_set_path + file) for file in ref_data_set]

    # results : list of bool. True if correct class is found, for each sample in test_data_set
    res = []

    with open("log.txt", "a") as log_file:
        for i_test, mat_test in enumerate(mat_test_data_set):
            v_dist = []
            for i_ref, mat_ref in enumerate(mat_ref_data_set):
                v_dist.append(process_distance(mat_test, mat_ref, wd=wd, wh=wh, wv=wv)[1])

            # v_dist = [process_distance(mat_test, mat_ref, wd=wd, wh=wh, wv=wv)[1] for mat_ref in mat_ref_data_set]
            best_k_ind = np.argsort(v_dist)[:k]
            best_k_ref = [ref_data_set[ind] for ind in best_k_ind]

            classes = [int(re.search(regex, ref).group(1)) for ref in best_k_ref]
            count = [classes.count(i) for i in range(10)]
            main_class = [i for i in range(10) if count[i]==max(count)][0]

            test_class = int(re.search(regex, test_data_set[i_test]).group(1))
            res.append(True if test_class == main_class else False)

            print(test_data_set[i_test], " --> ", main_class)
            print(best_k_ref)

            log_file.write(test_data_set[i_test] + " --> " + str(main_class) + "\n")
            log_file.write(best_k_ref.__repr__() + "\n")

    return res


def classify(k, test_proportion=0.2, data_set_path="output/", wv=1.0, wh=1.0, wd=1.0):
    """
    Classify with k nearest neighbours method.

    :param k: [int] number of neighbours to consider
    :param test_proportion: [float] proportion of test data
    :param data_set_path: [str] directory name in which .npy files are
    :param wd: [float] value for wd coefficient
    :param wv: [float] value for wv coefficient
    :param wh: [float] value for wh coefficient
    :return: [list of bool] list of booleans. True if a test data is correctly classified.
    """
    data_set = get_data_set(data_set_path)

    # proportion = 1
    # for key in data_set.keys():
        # data_set[key] = data_set[key][:int(proportion * len(data_set[key]))]

    test_data_set = []
    ref_data_set = []

    for number in data_set.keys():
        thr_index = int(test_proportion * len(data_set[number]))

        test_data_set += (data_set[number])[:thr_index]
        ref_data_set += (data_set[number])[thr_index:]

    random.shuffle(test_data_set)

    with open("log.txt", "w") as log_file:
        log_file.write("test data set : " + str(len(test_data_set)) + "\n")
        log_file.write("ref data set  : " + str(len(ref_data_set)) + "\n")
        log_file.write("wv  : " + str(wv) + "\n")
        log_file.write("wd  : " + str(wd) + "\n")
        log_file.write("wh  : " + str(wh) + "\n")

    print("test data set : ", len(test_data_set))
    print("ref data set  : ", len(ref_data_set))
    return k_nearest_neighbours(k, test_data_set, ref_data_set, wv=wv, wh=wh, wd=wd)