import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.linear_model import LinearRegression
from math import sqrt

import utils

x, y = utils.load_data_set()

n_training_samples = int(np.floor(0.8 * len(x)))
n_tests = int(np.floor(0.2 * len(x))) 

linear_regression = LinearRegression()



n_rounds = 100

figure = False
r_2 = []
rmse = []

for r in range(0, n_rounds):
    max_i = random.randint(0, len(x) - n_training_samples)

    training_range = np.random.permutation(range(0 + max_i, n_training_samples + max_i))

    x_training = x[training_range]
    y_training = y[training_range]

    linear_regression.fit(x_training, y_training)

    success_sum = 0

    tests_range = list(set(list(range(len(x)))) - set(training_range))

    quadratic_error = 0
    quadratic_error_sum = 0
    dev_quadratic_error_sum = 0

    x_test = x[tests_range]
    y_test = y[tests_range]

    mean = np.mean(y_test)

    y_predicted = []

    pred = linear_regression.predict(x_test)

    for test in range(0, len(x_test)):
        y_predicted.append(pred[test])

        error = y_test[test] - pred[test]

        quadratic_error = quadratic_error + (error**2)

        quadratic_error_sum = quadratic_error_sum + error**2
        dev_quadratic_error_sum = dev_quadratic_error_sum + (y_test[test] - mean)**2

    r_2.append(1 - quadratic_error_sum/dev_quadratic_error_sum)

    print("%.2f" % (1 - quadratic_error_sum/dev_quadratic_error_sum))
    
    rmse.append(dev_quadratic_error_sum)


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
print(sqrt(np.sum(rmse)))