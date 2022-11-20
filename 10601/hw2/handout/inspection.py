import sys
import numpy as np
import math


def read_data(file_path):
    """
    read data from files and cut the headers
    :param file_path: file name
    :return: data, without header
    """
    data = np.genfromtxt(file_path, delimiter="\t", dtype=None, encoding=None)
    data = data[1:]
    return data


def get_entropy(data):
    """
    calculate root entropy H(Y) (before splitting)
    Args: data:
    Returns:entropy
    """
    label_1 = data[0, -1]

    count_1 = 0
    count_2 = 0
    for label in data[:, -1]:
        if label == label_1:
            count_1 += 1
        else:
            count_2 += 1

    entropy1 = -(count_1 / len(data[:, -1])) * math.log(count_1 / len(data[:, -1]), 2)
    entropy2 = -(count_2 / len(data[:, -1])) * math.log(count_2 / len(data[:, -1]), 2)
    entropy = entropy2 + entropy1

    return entropy


def majority_vote(data):
    """
    ID majority element:
    if sequentially two elements are the same, both are kept.
    else, both are eliminated.
    the majority will remain
    :param data: one attribute of the point
    :return: the majority element
    """
    count = 0
    majority = None
    for label in data[:, -1]:
        if count == 0:
            majority = label
        if label == majority:
            count += 1
        else:
            count -= 1

    """in case of a tie"""
    if count == 0:
        if data[1, -1] == 'democrat' or data[1, -1] == 'republican':
            majority = 'republican'
        else:
            majority = 'A'

    return majority


def get_error(data, majority):
    """
    calculate the error rate
    :param data
    :param majority: the majority of training data
    :return:
    """
    labels = data[:, -1]
    count = 0
    for prediction in labels:
        if prediction != majority:
            count += 1
        else:
            continue
    error_rate = count / len(labels)

    return error_rate


if __name__ == '__main__':

    train_input = sys.argv[1]
    small_inspect = sys.argv[2]

    data = read_data(train_input)
    entropy = get_entropy(data)
    majority = majority_vote(data)
    error = get_error(data, majority)

    with open(small_inspect, 'w') as f:
        f.write("entropy: " + str(entropy) + "\n")
        f.write("error: " + str(error))


