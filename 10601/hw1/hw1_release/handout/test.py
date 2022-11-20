import numpy as np

f = "C:/Users/Zheyi/Documents/CMU/10601/hw1/hw1_release/handout/politicians_test.tsv"
data = np.genfromtxt(f, delimiter="\t", dtype=None, encoding=None)
data = data[1:]

# count = 0
# majority = None
# for label in data[:, -1]:
#     if count == 0:
#         majority = label
#         count += 1
#     else:
#         if label == majority:
#             count += 1
#         else:
#             count -= 1

label_1 = data[0, -1]
for label in data[:, -1]:
    if label != label_1:
        label_2 = label

print(label_1, label_2)

count_1 = 0
count_2 = 0
for label in data[:, -1]:
    if label == label_1:
        count_1 += 1
    else:
        count_2 += 1
if count_1 > count_2:
    majority = label_1
else:
    majority = label_2

print(count_1, count_2)
print(majority)

# test返回的label只取决于train

labels = data[:, -1]
count = 0
for prediction in labels:
    if prediction != majority:
        count += 1

print(count)

error_rate = count / len(labels)

print(error_rate)

