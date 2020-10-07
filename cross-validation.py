import numpy as np
def sigm(z):
    """
    Computes the sigmoid function

    :type z: float
    :rtype: float
    """
    return 1.0 / (1.0 + np.exp(-z))

def compute_grad(w, x, y):
    """
    Computes gradient of LL for logistic regression

    :type w: 1D np array of weights
    :type x: 2D np array of features where len(w) == len(x[0])
    :type y: 1D np array of labels where len(x) == len(y)
    :rtype: 1D numpy array
    """
    # number of features: d
    d = w.shape[0]
    # number of samples: n
    n = y.shape[0]
    # Initialize gradient as a d-dimensional array
    grad = np.zeros(d)

    for i in range(n):
        grad = grad - ((1.0 / (1.0 + np.exp(-1 * np.dot(w, x[i])))) - y[i]) * x[i]

    return grad

def gd_single_epoch(w, x, y, step):
    """
    Updates the weight vector by processing the entire training data once

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: 1D numpy array of weights
    """
    return w + step * compute_grad(w, x, y)

def gd(x, y, stepsize):
    """
    Iteratively optimizes the objective function by first
    initializing the weight vector with zeros and then
    iteratively updates the weight vector by going through
    the trianing data num_epoch_for_train(global var) times

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: 1D numpy array of weights
    """
    # Initialize as a d-dimensional vector.
    w = np.zeros(x.shape[1])
    # Update the weight vector through num_epoch_for_train iterations.
    for epoch in range(num_epoch_for_train):
        w = gd_single_epoch(w, x, y, stepsize)

    return w

def predict(w, x):
    """
    Makes a binary decision {0,1} based the weight vector
    and the input features

    :type w: 1D numpy array of weights
    :type x: 1D numpy array of features of a single data point
    :rtype: integer {0,1}
    """
    if sigm(np.dot(w, x)) > 0.5:
        return 1
    else:
        return 0

def accuracy(w, x, y):
    """
    Calculates the proportion of correctly predicted results to the total

    :type w: 1D numpy array of weights
    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as a proportion of correct labels to the total
    """
    num_correct = 0
    for i in range(x.shape[0]):
        if predict(w, x[i]) == y[i]:
            num_correct += 1

    return float(num_correct) / x.shape[0]

def five_fold_cross_validation_avg_accuracy(x, y, stepsize):
    """
    Measures the 5 fold cross validation average accuracy
    Partition the data into five equal size sets like
    |-----|-----|-----|-----|
    For all 5 choose 1 permutations, train on 4, test on 1.

    Compute the average accuracy using the accuracy function
    you wrote.

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :type stepsize: float
    :rtype: float as average accuracy across the 5 folds
    """
    # Split into 5 subsets
    x_subsets = np.split(x, 5)
    y_subsets = np.split(y, 5)
    # Sum of accuracies
    sum_accuracy = 0.0
    for i in range(5):
        # Set up training set and test set
        x_test_current = x_subsets[i]
        y_test_current = y_subsets[i]
        has_value = False
        x_train_current = None
        y_train_current = None
        for j in range(5):
            if i != j:
                if not has_value:
                    x_train_current = x_subsets[j]
                    y_train_current = y_subsets[j]
                    has_value = True
                else:
                    x_train_current = np.concatenate((x_train_current, x_subsets[j]))
                    y_train_current = np.concatenate((y_train_current, y_subsets[j]))

        # Training
        w = gd(x_train_current, y_train_current, stepsize)
        # Testing
        sum_accuracy = sum_accuracy + accuracy(w, x_test_current, y_test_current)

    return sum_accuracy / 5

def tune(x, y):
    """
    Optimizes the stepsize by calculating five_fold_cross_validation_avg_accuracy
    with 10 different stepsizes from 0.001, 0.002,...,0.01 in intervals of 0.001 and
    output the stepsize with the highest accuracy

    For comparison:
    If two accuracies are equal, pick the lower stepsize.

    NOTE: For best practices, we should be using Nested Cross-Validation for
    hyper-parameter search. Without Nested Cross-Validation, we bias the model to the
    data. We will not implement nested cross-validation for now. You can experiment with
    it yourself.
    See: http://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

    :type x: 2D numpy array of features where len(w) == len(x[0])
    :type y: 1D numpy array of labels where len(x) == len(y)
    :rtype: float as best stepsize
    """
    stepsizes = np.linspace(0.001, 0.01, num=10)
    max_accuracy = -1.0
    best_stepsize = None
    for stepsize in stepsizes:
        current_accuracy = five_fold_cross_validation_avg_accuracy(x, y, stepsize)
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_stepsize = stepsize

    return best_stepsize

w_single_epoch = gd_single_epoch(np.zeros(len(x_train[0])), x_train, y_train, default_stepsize)
w_optimized = gd(x_train, y_train, default_stepsize)
y_predictions = np.fromiter((predict(w_optimized, xi) for xi in x_test), x_test.dtype)
five_fold_average_accuracy = five_fold_cross_validation_avg_accuracy(x_train, y_train, default_stepsize)
tuned_stepsize = tune(x_train, y_train)