import numpy as np
import math
import operator
import matplotlib as plt

file_path = 'C:\\Users\\Zheyi\\Documents\\CMU\\10601\\hw2\\handout\\small_test.tsv'
dataset = np.genfromtxt(file_path, delimiter="\t", dtype=None, encoding=None)

data = []
for example in dataset:
    example = list(example)
    data.append(example)

headers = data[0]
data = data[1:]


def cal_ent(data):
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
    ret_data = []
    for feat_vec in data:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data.append(reduced_feat_vec)
    return ret_data


def chose_feature(data):
    num_features = len(data[0]) - 1
    base_ent = cal_ent(data)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data]
        unique_values = set(feat_list)
        new_ent = 0
        for value in unique_values:
            sub_data = split_data(data, i, value)
            prob = len(sub_data) / float(len(data))
            new_ent += prob * cal_ent(sub_data)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(classlist):
    class_count = {}
    for vote in classlist:
        if vote not in class_count.keys():
            class_count[vote] = 0
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, headers):
    class_list = [example[-1] for example in data]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return majority_cnt(class_list)
    best_feat = chose_feature(data)
    best_feat_header = headers[best_feat]
    tree = {best_feat_header: {}}
    del(headers[best_feat])
    feat_values = [example[best_feat] for example in data]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_headers = headers[:]
        tree[best_feat_header][value] = create_tree(split_data(data, best_feat, value), sub_headers)
    return tree


tree = create_tree(data, headers)
print(tree)


def predict(tree, test_data):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


