import sys
import numpy as np


def read_data(f):
    """
    read data from files and cut the headers (for train and test)
    :param f: file name
    :return: data
    """
    data = np.genfromtxt(f, delimiter="\t", dtype=None, encoding=None)
    data = data[1:]
    return data


def train_model(data):
    """
    ID majority element:
    if sequentially two elements are the same, both are kept.
    else, both are eliminated.
    the majority will remain
    :param data: one attribute of the point
    :return: the majority element
    """
    count = 0
    candidate = None
    for label in data[:, -1]:
        if count == 0:
            candidate = label
        if label == candidate:
            count += 1
        else:
            count -= 1

    """in case of a tie"""
    if count == 0:
        if data[1, -1] == 'democrat' or data[1, -1] == 'republican':
            candidate = 'republican'
        else:
            candidate = 'A'

    return candidate


def make_prediction(candidate):
    """
    predict: the prediction is the majority of training labels
    :param candidate: the majority of training labels
    :return: majority
    """
    majority = candidate
    return majority


def evaluate(data, majority):
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
    test_input = sys.argv[2]
    train_labels = sys.argv[3]
    test_labels = sys.argv[4]
    metrics = sys.argv[5]

    data_train = read_data(train_input)
    data_test = read_data(test_input)
    majority = train_model(data_train)

    # make prediction for training data
    results_train = []
    for i in range(len(data_train)):
        result = make_prediction(majority)
        results_train.append(result)
    with open(train_labels, 'w') as f:
        for result in results_train:
            f.write(result + "\n")

    # make prediction for test data
    results_test = []
    for i in range(len(data_test)):
        result = make_prediction(majority)
        results_test.append(result)
    with open(test_labels, 'w') as f:
        for result in results_test:
            f.write(result + "\n")

    error_train = evaluate(data_train, majority)
    error_train = format(error_train, '.6f')

    error_test = evaluate(data_test, majority)
    error_test = format(error_test, '.6f')

    with open(metrics, 'w') as f:
        f.write("error(train): " + error_train + "\n")
        f.write("error(test): " + error_test)
