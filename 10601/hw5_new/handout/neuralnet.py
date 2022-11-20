import numpy as np
import sys


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    labels = data[:, 0].astype(int)
    input_x = np.hstack((np.ones((len(data), 1)), data))  # add bias term
    input_x = np.delete(input_x, 1, axis=1)  # delete label column

    return input_x, labels


def label01(label):
    label_01 = np.zeros(10)
    label_01[label] = 1
    return label_01


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


def ave_cross_entropy(y_hat, y):
    # print(y_hat[10])
    # print(-np.dot(y.T, np.log(y_hat)).shape)
    summ = - np.sum(y * np.log(y_hat))
    return summ / len(y)


def forward(x, alpha, beta):  # x is one example, y is the corresponding label
    a = np.dot(alpha, x)
    a = a.reshape(-1, 1)
    z_star = sigmoid(a)
    z = np.vstack((np.array([1]), z_star))  # add z_0, column vector, M+1 * 1
    b = np.dot(beta, z)
    y_hat = softmax(b)  # row vector, 10 * 1
    return z, y_hat


def backward(beta, y_hat, y, z, example):
    y_hat = y_hat.reshape(-1, 1)
    y = y.reshape(-1, 1)
    g_beta = (y_hat - y).reshape(-1, 1) * z.T

    beta_star = np.delete(beta, 0, axis=1)

    z_star = np.delete(z.T, 0, axis=1).squeeze()
    g_z = np.dot(beta_star.T, y_hat.reshape(-1, 1) - y.reshape(-1, 1))
    g_a = g_z * np.multiply(z_star.reshape(-1, 1), (1 - z_star).reshape(-1, 1))
    g_alpha = np.multiply(g_a.reshape(-1, 1), example.reshape(-1, 1).squeeze())

    return g_alpha, g_beta


def train(train_input, train_label_in, val_input, val_labels, init_flag, num_hidden_units, num_epochs, learning_rate):
    D = num_hidden_units
    M = len(train_input[0]) - 1
    epsilon = 1e-5
    alpha, beta = initial_weight(init_flag, D, M)
    s_alpha = np.zeros((D, M + 1))
    s_beta = np.zeros((10, D + 1))
    cross_en_train_list = []
    cross_en_val_list = []

    for i in range(num_epochs):
        train_data, train_labels = shuffle(train_input, train_label_in, i)
        # print(train_data)
        for j in range(len(train_data)):
            z, y_hat = forward(train_data[j], alpha, beta)
            y = label01(train_labels[j])
            g_alpha, g_beta = backward(beta, y_hat, y, z, train_data[j])

            s_alpha += g_alpha ** 2
            s_beta += g_beta ** 2
            # print(s_beta)
            g_alpha_prime = g_alpha / (np.sqrt(s_alpha + epsilon))
            grad_beta_prime = g_beta / (np.sqrt(s_beta + epsilon))
            alpha = alpha - (learning_rate * g_alpha_prime)
            beta = beta - (learning_rate * grad_beta_prime)
        # predict & cross entropy
        labels_pred_train = []
        y_hat_train = []
        for example in train_data:
            y_hat_train.append(forward(example, alpha, beta)[1])
            labels_pred_train.append(np.argmax(forward(example, alpha, beta)[1]))
        y_hat_train = np.array(y_hat_train)
        labels_pred_train = np.array(labels_pred_train)

        label_01_train = np.zeros((len(train_labels), 10))
        for j in range(len(train_labels)):
            label = train_labels[j]
            label_01_train[j][label] = 1

        y_hat_train = y_hat_train.squeeze()
        cross_en_train = ave_cross_entropy(y_hat_train, label_01_train)
        cross_en_train_list.append(cross_en_train)

        labels_pred_val = []
        y_hat_val = []
        for example in val_input:
            y_hat_val.append(forward(example, alpha, beta)[1])
            labels_pred_val.append(np.argmax(forward(example, alpha, beta)[1]))
        y_hat_val = np.array(y_hat_val)
        labels_pred_val = np.array(labels_pred_val)

        y_hat_val = y_hat_val.squeeze()
        label_01_val = np.zeros((len(val_labels), 10))
        for j in range(len(val_labels)):
            label = val_labels[j]
            label_01_val[j][label] = 1
        cross_en_val = ave_cross_entropy(y_hat_val, label_01_val)
        cross_en_val_list.append(cross_en_val)
        # error coount
        error_trian = 0
        for k in range(len(labels_pred_train)):
            if labels_pred_train[k] != train_labels[k]:
                error_trian += 1
        error_trian /=  len(labels_pred_train)

        error_val = 0
        for k in range(len(labels_pred_val)):
            if labels_pred_val[k] != val_labels[k]:
                error_val += 1
        error_val /= len(labels_pred_val)
    return error_trian, error_val, cross_en_train_list, cross_en_val_list, labels_pred_train, labels_pred_val


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

    error_trian, error_val, cross_en_train_list, cross_en_val_list, labels_pred_train, labels_pred_val = \
        train(train_input_x, train_labels, val_input_x, val_labels, init_flag, D, num_epoch, learning_rate)

    with open(metrics_out, 'w') as f:
        for i in range(len(cross_en_train_list)):
            f.write('epoch={} crossentropy(train): {}\n'.format(i + 1, cross_en_train_list[i]))
            f.write('epoch={} crossentropy(validation): {}\n'.format(i + 1, cross_en_val_list[i]))
        f.write('error(train): {}\n'.format(error_trian))
        f.write('error(validation): {}\n'.format(error_val))

        with open(train_out, 'w') as f:
            for label in labels_pred_train:
                f.write('{}\n'.format(label))
        with open(valid_out, 'w') as f:
            for label in labels_pred_val:
                f.write('{}\n'.format(label))



