#python3

#Madeline using the LMSRule for training

import numpy as np
import matplotlib.pyplot as plt
import random
import math

import utils

def lmsrule(w, x, y, epochs, learning_rate):
    for i in range(0, epochs):
        for j in range(0, len(x)):
            y_pred = w @ x[j]

            #y_pred = (1.0)/(1.0 + math.exp(-y_pred))

            err = y[j] - y_pred

            x_norm = x[j]/(x[j] @ x[j])

            delta = err[:,None] @ x_norm[:,None].T

            w = w + learning_rate*delta

    return w

x, y = utils.load_data_set()

n_training_samples = int(np.floor(0.8 * len(x)))
n_tests = int(np.floor(0.2 * len(x))) 

n_rounds = 50
epochs = 50
learning_rate = 0.0005

success_rate_sum = 0

highest_success_rate = 0
lowest_success_rate = 1

figure = False
r_2 = []

for r in range(0, n_rounds):
    max_i = random.randint(0, len(x) - n_training_samples)

    training_range = np.random.permutation(range(0 + max_i, n_training_samples + max_i))

    x_training = x[training_range]
    y_training = y[training_range]
    #
    w_init = 0.1 * np.random.rand(len(y_training[0]), len(x_training[0]))

    w = lmsrule(w_init, x_training, y_training, epochs, learning_rate)

    success_sum = 0

    tests_range = list(set(list(range(len(x)))) - set(training_range))

    quadratic_error = 0
    quadratic_error_sum = 0
    dev_quadratic_error_sum = 0

    x_test = x[tests_range]
    y_test = y[tests_range]

    mean = np.mean(y_test)

    y_predicted = []

    for test in range(0, len(x_test)):
        pred = w @ np.matrix.flatten(x_test[test])

        y_predicted.append(pred)

        error = y_test[test] - pred

        quadratic_error = quadratic_error + (error**2)

        quadratic_error_sum = quadratic_error_sum + error**2
        dev_quadratic_error_sum = dev_quadratic_error_sum + (y_test[test] - mean)**2

    r_2.append(1 - quadratic_error_sum/dev_quadratic_error_sum)

    if figure:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, len(y_test), 1))
        #ax.set_yticks(np.arange(-1, 1., 0.1))
        #plt.scatter(x, y)
        plt.grid()
        plt.plot([i for i in range(len(y_test))], y_test, 'r+')
        plt.plot([i for i in range(len(y_test))], y_predicted, 'bo')
        plt.show()

print(np.mean(r_2))
