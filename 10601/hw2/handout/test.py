import numpy as np
import math


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


def get_entropy(data):
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
    mutual_info = entropy - con_entropy
    return mutual_info


def get_best_feature(data):
    num_attr = len(data) - 1
    mutual = []
    for i in range(num_attr):
        entropy = get_entropy(data)
        con_entropy = get_conditional_entropy(data, i)
        mutual_info = get_mutual_information(entropy, con_entropy)
        mutual.append(mutual_info)
    feature_index = mutual.index(max(mutual))

    return feature_index


def cal_entropy(data):
    """
    calculate entropy
    Args:
        data:
    Returns: entropy
    """
    num_entries = len(data)
    label_counts = {}
    for feat_vec in data:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
        ent = 0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            ent -= prob * math.log(prob, 2)
    return ent


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


def chose_feature(data):
    """
    choose feature to split
    Args:
        data:
    Returns: best feature index
    """
    num_features = len(data[0]) - 1
    base_ent = cal_entropy(data)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data]
        unique_values = set(feat_list)
        new_ent = 0
        for value in unique_values:
            sub_data = split_data(data, i, value)
            prob = len(sub_data) / float(len(data))
            new_ent += prob * cal_entropy(sub_data)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


train_title, train_data = store_datas(open(r'C:\Users\Zheyi\Documents\CMU\10601\hw2\handout\small_test.tsv'))
data = np.genfromtxt(r'C:\Users\Zheyi\Documents\CMU\10601\hw2\handout\small_test.tsv', delimiter="\t", dtype=None, encoding=None)
data = data[1:]

print(chose_feature(train_data))
print(get_best_feature(data))



