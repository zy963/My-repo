import numpy as np
import sys


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    labels = data[:, 0].astype(int)
    input_x = np.hstack((np.ones((len(data), 1)), data))  # add bias term
    input_x = np.delete(input_x, 1, axis=1)  # delete label column
    return input_x, labels


def shuffle(X, y, epoch):
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def random_init(x, y):
    np.random.seed(np.prod((x, y)))
    weights = np.random.uniform(-0.1, 0.1, (x, y))
    weights[:, 0] = 0.0
    return weights


def zero_init(x, y):
    weights = np.zeros((x, y))
    return weights


def initial_weight(init_flag, D, M):  # D: number of hidden units; M: length of each example
    if init_flag == 1:
        alpha = random_init(D, M + 1)
        beta = random_init(10, D + 1)
    elif init_flag == 2:
        alpha = zero_init(D, M + 1)
        beta = zero_init(10, D + 1)
    return alpha, beta


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


def cross_entropy(y_hat, y):
    # print(y_hat[10])
    # print(-np.dot(y.T, np.log(y_hat)).shape)
    sum1 = - np.sum(y * np.log(y_hat))
    # print(sum1)
    return sum1/len(y)
    # return -np.dot(y.T, np.log(y_hat))


def forward(x, alpha, beta):  # x is one example, y is the corresponding label
    a = np.dot(alpha, x)
    a = a.reshape(-1, 1)
    z_star = sigmoid(a)
    z = np.vstack((np.array([1]), z_star))  # add z_0, column vector, M+1 * 1
    b = np.dot(beta, z)
    y_hat = softmax(b)  # row vector, 10 * 1
    return z, y_hat


def g_beta(y_hat, y, z):  # g_beta = [y_hat - y] * z.T
    y_hat = y_hat.reshape(-1, 1)
    # print(y_hat.shape)
    y = y.reshape(-1, 1)
    # print(y.shape) # 5000,1
    return (y_hat - y).reshape(-1, 1) * z.T


def g_alpha(beta, y_hat, y, z, example):
    # g_alpha = g_a * example.T
    # g_a = g_z * [z] * [1 - z]
    # g_z = beta_without_bias.T * [y_hat - y]
    beta_star = np.delete(beta, 0, axis=1)
    z_star = np.delete(z.T, 0, axis=1).squeeze()
    g_z = np.dot(beta_star.T, y_hat.reshape(-1, 1) - y.reshape(-1, 1))
    g_a = g_z * np.multiply(z_star.reshape(-1, 1), (1 - z_star).reshape(-1, 1))
    g_alpha = np.multiply(g_a.reshape(-1, 1), example.reshape(-1, 1).squeeze())
    return g_alpha


def one_hot_test(data, alpha, beta):
    y_pred = []
    y_hat = []
    for example in data:
        y_hat.append(forward(example, alpha, beta)[1])
        y_pred.append(np.argmax(forward(example, alpha, beta)[1]))
    # print(np.array(y_hat)[10])
    return np.array(y_hat), np.array(y_pred)


def sgd(train_input, labels, init_flag, num_hidden_units, num_epochs, learning_rate):
    D = num_hidden_units
    M = len(train_input[0]) - 1

    epsilon = 1e-5
    alpha, beta = initial_weight(init_flag, D, M)


    s_alpha = np.zeros((D, M + 1))
    s_beta = np.zeros((10, D + 1))

    for i in range(num_epochs):
        train_input_x, labels = shuffle(train_input, labels, i)
        label_01 = np.zeros((len(train_input_x), 10))
        for j in range(M + 1):
            label = labels[j]
            label_01[j][label] = 1

            z, y_hat = forward(train_input_x[j], alpha, beta)
            g_bet = g_beta(y_hat, label_01[j], z)
            g_alph = g_alpha(beta, y_hat, label_01[j], z, train_input_x[j])
            s_alpha += g_alph ** 2
            s_beta += g_bet ** 2
            alpha -= learning_rate * g_alph / (np.sqrt(s_alpha + epsilon))
            beta -= learning_rate * g_bet / (np.sqrt(s_beta + epsilon))
    # print(alpha)
    return alpha, beta, label_01


def predict_eval(input_x, alpha, beta, labels, label_01):
    y_hats, labels_pred = one_hot_test(input_x, alpha, beta)
    ave_cross_en = []
    label_01 = label_01.squeeze()
    y_hats = y_hats.squeeze()
    # print(y_hats.shape)
    cross_en = cross_entropy(y_hats, label_01)
    ave_cross_en.append(cross_en)

    error = 0
    for i in range(len(labels)):
        if labels_pred[i] != labels[i]:
            error += 1
    # print(np.array(ave_cross_en).sum()/len(labels))
    return ave_cross_en, error/len(labels), labels_pred


if __name__ == "__main__":
    train_input = sys.argv[1]
    valid_input = sys.argv[2]
    train_out = sys.argv[3]
    valid_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    train_input_x, train_labels = load_data(train_input)
    val_input_x, val_labels = load_data(valid_input)

    D = hidden_units
    M = len(train_input_x[0]) - 1

    alpha_trained, beta_trained, label_01 = sgd(train_input_x, train_labels, init_flag, D, num_epoch, learning_rate)
    ave_cross_en_train, error_train, labels_pred_train = predict_eval(train_input_x, alpha_trained, beta_trained, train_labels, label_01)

    label_01_val = np.zeros((len(val_labels), 10)) # 100,10
    # print(val_labels.shape)
    for i in range(len(val_labels)):
        la = val_labels[i]
        label_01_val[i][la] = 1

    ave_cross_en_val, error_val, labels_pred_val = predict_eval(val_input_x, alpha_trained, beta_trained, val_labels, label_01_val)

    with open(metrics_out, 'w') as f:
        for i in range(len(ave_cross_en_train)):
            f.write('epoch={} crossentropy(train): {}\n'.format(i + 1, ave_cross_en_train[i]))
            f.write('epoch={} crossentropy(validation): {}\n'.format(i + 1, ave_cross_en_val[i]))
        f.write('error(train): {}\n'.format(error_train))
        f.write('error(validation): {}\n'.format(error_train))

        with open(train_out, 'w') as f:
            for label in labels_pred_train:
                f.write('{}\n'.format(label))
        with open(valid_out, 'w') as f:
            for label in labels_pred_val:
                f.write('{}\n'.format(label))




