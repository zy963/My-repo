import sys
import numpy as np
import math
import operator
from collections import Counter


def store_datas(in_file):
    """
    read data from file
    Args:
        in_file: file name
    Returns: data titles and data, type = list
    """
    res = []
    first = True
    titles = []
    for line in in_file:
        line = line.replace('\n', '')
        data = line.split('\t')
        if first:
            titles = data
            first = False
            continue
        cur_data = data
        res.append(list(cur_data))
    return titles, res


def split_data(data, axis, value):
    """
    split data base on attribute value
    Args:
        data:
        axis: column index
        value: attribute value
    Returns: rest of data
    """
    ret_data = []
    for feat_vec in data:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data.append(reduced_feat_vec)
    return ret_data


def get_entropy(data):
    """
    calculate entropy
    Args:
        data:
    Returns: entropy
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


def get_conditional_entropy(data, feature_index):
    """
    calculate conditional entropy
    Args:
        data: same with calculate entropy data
        feature_index: column index
    Returns: conditional entropy
    """
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0

    feature_values = data[:, [feature_index]]
    unique_values = np.unique(feature_values)
    labels = data[:, [-1]]
    unique_labels = np.unique(labels)

    for current in data:
        if current[feature_index] == unique_values[0] and current[-1] == unique_labels[0]:
            count1 += 1
        elif current[feature_index] != unique_values[0] and current[-1] == unique_labels[0]:
            count2 += 1
        elif current[feature_index] == unique_values[0] and current[-1] != unique_labels[0]:
            count3 += 1
        elif current[feature_index] != unique_values[0] and current[-1] != unique_labels[0]:
            count4 += 1

    temp1 = (count1 / (count1 + count3)) * math.log(count1 / (count1 + count3), 2) if count1 != 0 else 0
    temp2 = (count2 / (count2 + count4)) * math.log(count2 / (count2 + count4), 2) if count2 != 0 else 0
    temp3 = (count3 / (count1 + count3)) * math.log(count3 / (count1 + count3), 2) if count3 != 0 else 0
    temp4 = (count4 / (count2 + count4)) * math.log(count4 / (count2 + count4), 2) if count4 != 0 else 0
    con_entropy = -((count1 + count3) / len(data)) * (temp1 + temp3) - ((count2 + count4) / len(data)) * (temp2 + temp4)

    return con_entropy


def get_mutual_information(entropy, con_entropy):
    """
    get mutual information based on entropy and conditional entropy
    Args:
        entropy:
        con_entropy:
    Returns: mutual information
    """
    mutual_info = entropy - con_entropy
    return mutual_info


def get_best_feature(data):
    """
    choose feature to split on (max mutual information)
    Args:
        data:
    Returns: feature index (column index)
    """
    num_attr = data.shape[1] - 1
    mutual = []
    for i in range(num_attr):
        entropy = get_entropy(data)
        con_entropy = get_conditional_entropy(data, i)
        mutual_info = get_mutual_information(entropy, con_entropy)
        mutual.append(mutual_info)
    feature_index = mutual.index(max(mutual))
    return feature_index


def majority(classlist):
    """
    majority vote
    Args:
        classlist:
    Returns: majority
    """
    class_count = Counter(classlist)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, headers, max_depth):
    """
    tree generation
    Args:
        data: full dataset
        headers: attribute names
    Returns: tree, embedded dict
    """
    class_list = []
    depth = 0
    for i in range(len(data)):
        class_list.append(data[i][-1])
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return majority(class_list)
    if depth >= max_depth:
        return majority(class_list)
    best_feat = get_best_feature(np.array(data))
    best_feat_header = headers[best_feat]
    tree = {best_feat_header: {}}
    del (headers[best_feat])
    feat_values = []
    for d in data:
        feat_values.append(d[best_feat])
    unique_values = set(feat_values)
    for v in unique_values:
        re = create_tree(split_data(data, best_feat, v), headers[:], max_depth - 1)
        tree[best_feat_header][v] = re
    return tree


def get_index(header, v):
    """
    get the index base on header and value
    Args:
        header:
        v:
    Returns: index
    """
    for j in range(len(header)):
        if header[j] == v:
            return j
    return -1



def predict(node, cur_d, header):
    """
    make prediction
    Args:
        node:
        cur_d: current data (row)
        header: column name
    Returns: node if it's a leaf, else: recursion
    """
    if type(node) == np.str_ or type(node) == str:
        return node
    for k, v in node.items():
        index = get_index(header, k)
        if cur_d[index] in v:
            return predict(v[cur_d[index]], cur_d, header)
        else:
            return v


def eval(roots, datas, header):
    """
    make prediction and calculate error rate
    Args:
        roots: global root
        datas:
        header:
    Returns: predictions: list; error rate
    """
    res = []
    err = 0
    for i, each_d in enumerate(datas):
        re = predict(roots, each_d, header)
        res.append(re)
        if re != each_d[-1]:
            err += 1
    return res, err / len(datas)


if __name__ == '__main__':
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    path = ''

    train_title, train_data = store_datas(open(train_file_name))
    test_title, test_data = store_datas(open(test_file_name))

    tree = create_tree(train_data, train_title.copy(), 3)
    test_res, test_error = eval(tree, test_data, test_title)
    train_res, train_error = eval(tree, train_data, train_title)

    with open(train_out, 'w') as f:
        for res in train_res:
            f.write(str(res) + "\n")
    with open(test_out, 'w') as f:
        for res in test_res:
            f.write(str(res) + "\n")

    train_error = str(train_error)
    test_error = str(test_error)
    with open(metrics_out, 'w') as f:
        f.write("error(train): " + train_error + "\n")
        f.write("error(test): " + test_error)



