import numpy as np
import sys

path_val = sys.argv[1]
path_index2word = sys.argv[2]
path_index2tag = sys.argv[3]
path_hmminit = sys.argv[4]
path_hmmemit = sys.argv[5]
path_hmmtrans = sys.argv[6]
path_predicted = sys.argv[7]
path_metric = sys.argv[8]

pi = np.genfromtxt(path_hmminit, delimiter='\n')
B = np.genfromtxt(path_hmmtrans)
A = np.genfromtxt(path_hmmemit)
dict_word = {}
n1 = int(0)
with open(path_index2word, 'r') as f:
    for line in f.readlines():
        dict_word[line.rstrip('\n')] = n1
        n1 += 1
'''
dict_tag = {}
n2 = int(0)
with open(path_index2tag, 'r') as f:
    for line in f.readlines():
        dict_tag[line.rstrip('\n')] = n2
        n2 += 1
'''
tag = np.genfromtxt(path_index2tag, delimiter='\n', dtype='str')

with open(path_val, 'r') as f:
    d_val = f.read().split('\n\n')
val_seqs = []
for sequence in d_val:
    val_seqs.append(sequence.split('\n'))

word_seqs = []
tag_seqs = []
num_word = 0
for seq in val_seqs:
    seq_w = []
    seq_t = []
    for pair in range(len(seq)):
        ww, tt = seq[pair].split('\t')
        seq_w.append(ww)
        seq_t.append(tt)
        num_word += 1
    word_seqs.append(seq_w)
    tag_seqs.append(seq_t)


def forward(word_seq):
    num_word = len(word_seq)
    num_tag = len(tag)
    alpha = np.zeros(shape=(num_word, num_tag))
    for i in range(num_word):
        if i == 0:
            alpha[0] = np.log(pi) + np.log(A[:, dict_word[word_seq[0]]])
        else:
            for j in range(num_tag):
                v = np.zeros(num_tag)
                for k in range(num_tag):
                    v[k] = alpha[i - 1, k] + np.log(B[k, j])
                m = np.max(v)
                log_sum_exp = m + np.log(np.sum(np.exp(v - m)))
                alpha[i, j] = np.log(A[j, dict_word[word_seq[i]]]) + log_sum_exp
    return alpha


def backward(word_seq):
    num_word = len(word_seq)
    num_tag = len(tag)
    beta = np.zeros(shape=(num_word, num_tag))
    for i in range(num_word):
        if i == 0:
            beta[num_word - 1] = np.zeros(num_tag)
        else:
            for j in range(num_tag):
                v = np.zeros(num_tag)
                for k in range(num_tag):
                    v[k] = beta[num_word - i, k] + np.log(B[j, k]) + np.log(A[k, dict_word[word_seq[num_word - i]]])
                m = np.max(v)
                log_sum_exp = m + np.log(np.sum(np.exp(v - m)))
                beta[num_word - i - 1, j] = log_sum_exp
    return beta


def forwardback(word_seq):
    alpha = forward(word_seq)
    beta = backward(word_seq)
    product = alpha + beta
    y_h = np.argmax(product, axis=1)
    yh_tag = tag[y_h]
    vv = np.max(alpha[-1])
    llh = vv + np.log(np.sum(np.exp(alpha[-1] - vv)))
    return yh_tag, llh


sum_llh = 0
count = 0
text = ''
for i in range(len(word_seqs)):
    tag_h, llh = forwardback(word_seqs[i])
    sum_llh += llh
    for j in range(len(tag_h)):
        text = text + word_seqs[i][j] + '\t' + tag_h[j] + '\n'
        if tag_seqs[i][j] == tag_h[j]:
            count += 1
    if i != (len(word_seqs) - 1):
        text = text + '\n'

ave_llh = sum_llh / len(word_seqs)
accuracy = count / num_word

with open(path_metric, 'w') as f:
    f.write(f'Average Log-Likelihood: {ave_llh}\nAccuracy: {accuracy}\n')

with open(path_predicted, 'w') as f:
    f.write(text)

# python forwardbackward.py validation.txt index_to_word.txt index_to_tag.txt hmminit.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt