import sys
import numpy as np
import math
import operator
from collections import Counter


def store_datas(in_file):
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


def cal_entropy(data):
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
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data.append(reduced_feat_vec)
    return ret_data


def chose_feature(data):
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


def majority(classlist):
    class_count = Counter(classlist)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, headers):
    class_list = []
    for i in range(len(data)):
        class_list.append(data[i][-1])
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return majority(class_list)
    best_feat = chose_feature(data)
    best_feat_header = headers[best_feat]
    tree = {best_feat_header: {}}
    del (headers[best_feat])
    feat_values = []
    for d in data:
        feat_values.append(d[best_feat])
    unique_values = set(feat_values)
    for v in unique_values:
        re = create_tree(split_data(data, best_feat, v), headers[:])
        tree[best_feat_header][v] = re
    return tree


def get_index(header, v):
    for j in range(len(header)):
        if header[j] == v:
            return j
    return -1


def predict(node, cur_d, header):
    if type(node) == np.str_ or type(node) == str:
        return node
    for k, v in node.items():
        index = get_index(header, k)
        return predict(v[cur_d[index]], cur_d, header)


def eval(roots, datas, header):
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

    tree = create_tree(train_data, train_title.copy())
    test_res, test_error = eval(tree, test_data, test_title)
    train_res, train_error = eval(tree, train_data, train_title)

    with open(train_out, 'w') as f:
        for res in train_res:
            f.write(res + "\n")
    with open(test_out, 'w') as f:
        for res in test_res:
            f.write(res + "\n")

    train_error = str(train_error)
    test_error = str(test_error)
    with open(metrics_out, 'w') as f:
        f.write("error(train): " + train_error + "\n")
        f.write("error(test): " + test_error)
