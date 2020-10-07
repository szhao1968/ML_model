import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    sigmoid function
    
    :param ndarray z
    """
    return 1.0/(1.0+np.exp(-z))

def predict(x, w, b):
    """
    Forward prediction of neural network
    
    :param ndarray x: num_feature x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype int: label index, starting from 1
    """

    for wl, bl in zip(w, b):
        x = sigmoid(np.dot(wl, x) + bl)

    return np.argmax(x) + 1

def accuracy(testing_data, testing_label, w, b):
    """
    Return the accuracy(0 to 1) of the model w, b on testing data
    
    :param ndarray testing_data: num_data x num_feature numpy array
    :param ndarray testing_label: num_data x 1 numpy array
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :rtype float: accuracy(0 to 1)
    """

    correct = 0.0
    num_feature = len(testing_data[0])
    for i in range(len(testing_data)):
        sample = np.zeros((num_feature, 1))
        sample[:, 0] = testing_data[i, :]
        y = predict(sample, w, b)
        if y == testing_label[i]:
            correct += 1

    return correct / len(testing_data)

def gradient(x, y, w, b):
    """
    Compute gradient using backpropagation
    
    :param ndarray x: num_feature x 1 numpy array
    :param ndarray y: num_label x 1 numpy array
    :rtype tuple: A tuple contains the delta/gradient of weights and bias (dw, db)
                dw and db should have same format as w and b correspondingly
    """

    z = [None] * len(w)
    a = [None] * len(w)

    for i in range(len(w)):
        if i == 0:
            z[i] = np.dot(w[0], x) + b[0]
        else:
            z[i] = np.dot(w[i], a[i - 1]) + b[i]
        a[i] = sigmoid(z[i])

    delta = [None] * len(w)
    for i in reversed(range(len(w))):
        if i == len(w) - 1: # Last layer
            error = a[len(w) - 1] - y # 3x1
            delta[i] = error * a[i] * (1 - a[i])
        else:
            delta[i] = np.dot(w[i + 1].T, delta[i + 1]) * a[i] * (1 - a[i])

    dw = [None] * len(w)
    db = [None] * len(w)
    for i in range(len(w)):
        if i == 0:
            dw[i] = np.dot(delta[i], x.T)
        else:
            dw[i] = np.dot(delta[i], a[i - 1].T)
        db[i] = delta[i]

    return dw, db


def single_epoch(w, b, training_data, training_label, eta, num_label):
    """
    Compute one epoch of batch gradient descent
    
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """

    num_data = len(training_data)
    num_feature = len(training_data[0])
    sum_dw = None
    sum_db = None
    for i in range(num_data):
        one_hot = np.zeros((num_label, 1))
        one_hot[training_label[i] - 1, 0] = 1
        sample = np.zeros((num_feature, 1))
        sample[:, 0] = training_data[i].T
        dw, db = gradient(sample, one_hot, w, b)

        if not sum_dw:
            sum_dw = dw.copy()
            sum_db = db.copy()
        else:
            for layer in range(len(w)):
                sum_dw[layer] += dw[layer]
                sum_db[layer] += db[layer]

    for layer in range(len(w)):
        w[layer] -= eta * sum_dw[layer] / num_data
        b[layer] -= eta * sum_db[layer] / num_data

    return w, b


def batch_gradient_descent(w, b, training_data, training_label, eta, num_label, num_epochs = 200, show_plot = False):
    """
    Train the NN model using batch gradient descent
    
    :param list w: follows the format of "weights" declared below
    :param list b: follows the format of "bias" declared below
    :param ndarray training_data: num_data x num_feature numpy array
    :param ndarray training_label: num_data x 1 numpy array
    :param float eta: step size
    :param int num_label: number of labels
    :rtype tuple: A tuple contains the updated weights and bias (w, b)
                w and b should have same format as they are pased in
    """
    train_acc = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    w_copy = np.copy(w)
    b_copy = np.copy(b)

    for i in range(num_epochs):
        train_acc[i] = accuracy(training_data, training_label, w_copy, b_copy)
        test_acc[i] = accuracy(testing_data, testing_label, w_copy, b_copy)
        # print 'start epoch {}, train acc {} test acc {}'.format(i, train_acc, test_acc)
        w_copy,b_copy =single_epoch(w_copy, b_copy, training_data, training_label, eta, num_label)

    if show_plot:
        plt.figure()
        train_line = plt.plot(np.array(range(1, num_epochs + 1)), train_acc, 'b-', label='Train Accuracy')
        test_line = plt.plot(np.array(range(1, num_epochs + 1)), test_acc, 'r-', label='Test Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('eta = %f' % eta)
        plt.show()

    return (w_copy,b_copy)

num_label = 3
num_feature = len(training_data[0])
num_hidden_nodes = 50 #50 is not the best parameter, but we fix it here
step_sizes = [0.3,3,10]

def deep_copy(init_weights, init_bias):
    w = []
    b = []
    for wl, bl in zip(init_weights, init_bias):
        w.append(wl.copy())
        b.append(bl.copy())
    return w, b

for step_size in step_sizes:
    w, b = deep_copy(init_weights, init_bias)
    w, b = batch_gradient_descent(
        w, b, training_data, training_label, step_size, num_label, 200, show_plot=True)

print('The plot of using learning rate 0.3 shows that there is a huge gap between test accuracy and training accuracy.')
print('The plto of using learning rate 10.0 shows the problem of overshooting since the step size is too large. From the plot, we can observe that the model is zig-zagging in its early stage.')

w, b = deep_copy(init_weights, init_bias)
weights, bias = batch_gradient_descent(
    w, b, training_data, training_label, 3.0, num_label, 200)
