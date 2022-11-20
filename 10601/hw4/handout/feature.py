import csv
import numpy as np
from collections import Counter
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file, MAX_WORD_LEN):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def model_1(dataset, dic):
    """
    feature engineering: bag of words
    Args:
        dataset: dataset, with first column being labels and second column being reviews
        dic: word dictionary

    Returns: an array, with len(dic) columns and len(review) rows

    """
    label = [x[0] for x in dataset]
    review = [x[1] for x in dataset]
    N = len(review)
    M = len(dic.keys())
    z = np.zeros((N, M+1))
    for i in range(N):
        z[i, 0] = label[i]
        for word in review[i].split(' '):
            if word in dic.keys():
                z[i, dic[word]+1] = 1
    return z


def model_2(dataset, word2vec_map):
    """
    feature engineering: word embeddings
    Args:
        dataset: dataset, with first column being labels and second column being reviews
        word2vec_map: word dictionary

    Returns:

    """
    result = []
    for label, review in dataset:
        count = 0
        vector = np.zeros(300)
        for word in review.split(" "):
            if word in word2vec_map:
                count += 1
                vector += word2vec_map[word]
        result.append(np.concatenate([[float(label)], vector / count], axis=0))

    return result


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    feature_dict_input = sys.argv[5]
    formatted_train_out = sys.argv[6]
    formatted_validation_out = sys.argv[7]
    formatted_test_out = sys.argv[8]
    feature_flag = sys.argv[9]

    VECTOR_LEN = 300  # Length of word2vec vector
    MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

    train_dataset = load_tsv_dataset(train_input)
    validation_dataset = load_tsv_dataset(validation_input)
    test_dataset = load_tsv_dataset(test_input)

    dic = load_dictionary(dict_input, MAX_WORD_LEN)
    word2vec_map = load_feature_dictionary(feature_dict_input)

    if feature_flag == '1':
        formatted_train_out = model_1(train_dataset, dic)
        formatted_validation_out = model_1(validation_dataset, dic)
        formatted_test_out = model_1(test_dataset, dic)
        np.savetxt(sys.argv[6], formatted_train_out, delimiter="\t", fmt="%s")
        np.savetxt(sys.argv[7], formatted_validation_out, delimiter="\t", fmt="%s")
        np.savetxt(sys.argv[8], formatted_test_out, delimiter="\t", fmt="%s")

    if feature_flag == '2':
        formatted_train_out = model_2(train_dataset, word2vec_map)
        formatted_validation_out = model_2(validation_dataset, word2vec_map)
        formatted_test_out = model_2(test_dataset, word2vec_map)
        np.savetxt(sys.argv[6], formatted_train_out, delimiter="\t", fmt="%s")
        np.savetxt(sys.argv[7], formatted_validation_out, delimiter="\t", fmt="%s")
        np.savetxt(sys.argv[8], formatted_test_out, delimiter="\t", fmt="%s")
