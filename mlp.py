import numpy as np
import matplotlib.pyplot as plt
import random
import math

import utils as utils

class mlp:
    def __init__(self, x_training, y_training):
        self.x_training = x_training
        self.y_training = y_training
        self.results = []

    def config(self, epochs, eta, mom, classification, n_o_neurons, n_h_layers, n_h_neurons, h_eta):
        self.epochs = epochs
        self.eta = eta
        self.mom = mom
        self.classification = classification
        self.n_o_neurons = n_o_neurons
        self.n_h_layers = n_h_layers
        self.n_h_neurons = n_h_neurons
        self.h_eta = h_eta

        self.hidden_layers = []

        for i in range(0, n_h_layers):
            if i == 0:
                w = 0.01 * np.random.rand(n_h_neurons[i], len(x_training[0]) + 1)
            else:
                w = 0.01 * np.random.rand(n_h_neurons[i], len(self.hidden_layers[i-1]))
            self.hidden_layers.append(w)

        self.output_layer = 0.01 * np.random.rand(n_o_neurons, n_h_neurons[n_h_layers - 1] + 1)

    def __sigmoide(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def __hyperbolic_tangent(self, x):
        return (1.0 - np.exp(-x))/(1.0 + np.exp(-x))

    def __activation(self, x):
        if self.classification:
            return self.__sigmoide(x)
        else:
            return self.__hyperbolic_tangent(x)

    def train(self, samples):
        #Store layers' current values 
        old_hidden_layers = []
        for i in range(0, len(self.hidden_layers)):
            old_hidden_layers.append(self.hidden_layers[i])

        old_output_layer = self.output_layer
        for i in range(0, self.epochs):
            quadratic_error = 0

            max_i = random.randint(0, len(self.x_training) - samples)

            training_range = np.random.permutation(range(0 + max_i, samples + max_i))

            for j in training_range:
                #Hidden layer neuron's activation
                u = []
                y = []

                #Calculates hidden layers
                for k in range(0, self.n_h_layers):
                    if k == 0:
                        u_k = self.hidden_layers[k] @ (np.insert(self.x_training[j], 0, 1))
                    else:
                        u_k = self.hidden_layers[k] @ y[k-1] 

                    u.append(u_k)
                    
                    y_k = []
                    for l in range(0, len(u_k)):
                        y_k.append(self.__activation(u_k[l]))

                    y.append(y_k)
                
                hidden_layer_output = y[len(self.hidden_layers) - 1]

                #Computes output layer
                o = self.output_layer @ np.array(np.insert(hidden_layer_output, 0, 1)).T
                for k in range(0, len(o)):
                    o[k] = self.__activation(o[k])

                error = y_training[j] - o
                quadratic_error = quadratic_error + 0.5*(pow(error, 2))

                #Output layer gradient
                output_d = []
                if self.classification:
                    output_d = np.multiply(error, np.multiply(o, 1 - o) + 0.05)
                else:
                    output_d = np.multiply(error, 1 - np.multiply(o, o) * 0.05)

                #Hidden layer gradient
                hidden_d = [None] * len(self.hidden_layers)
                for k in range(len(self.hidden_layers) - 1, -1, -1):
                    d = []
                    if self.classification:
                        if k == len(self.hidden_layers) - 1:
                            d = np.multiply(
                                        np.multiply(y[k], 1 - np.array(y[k])) + 0.05,
                                        np.transpose(self.output_layer[:, 1:]) @ output_d)
                        else:
                            d = np.multiply(
                                        np.multiply(y[k], 1 - np.array(y[k])) + 0.05,
                                        np.transpose(self.hidden_layers[k+1]) @ hidden_d[k+1]
                                    )
                    else:
                        if k == len(self.hidden_layers) - 1:
                            d = np.multiply(
                                        1 - np.multiply(np.array(y[k]), y[k]) * 0.05,
                                        np.transpose(self.output_layer[:, 1:]) @ output_d)
                        else:
                            d = np.multiply(
                                        1 - np.multiply(np.array(y[k]), y[k]) * 0.05,
                                        np.transpose(self.hidden_layers[k+1]) @ hidden_d[k+1]
                                    )

                    hidden_d[k] = d

                aux_output_layer = self.output_layer
                self.output_layer = (self.output_layer
                                    + self.eta
                                    * np.array(output_d[:, None]) @ np.array(np.insert(hidden_layer_output, 0, 1)[:, None]).T
                                    #+ mom * (self.output_layer - old_output_layer)
                                    )
                old_output_layer = aux_output_layer

                for k in range(0, len(self.hidden_layers)):
                    aux_hidden_layer = self.hidden_layers[k]
                    if k == 0:
                        self.hidden_layers[k] = (self.hidden_layers[k] 
                                                + self.h_eta[k]
                                                * np.array(hidden_d[k])[:, None] @ np.array(np.insert(self.x_training[j], 0, 1)[:, None]).T
                                               # + mom * (self.hidden_layers[k] - old_hidden_layers[k])
                                            )
                    else:
                        self.hidden_layers[k] = (self.hidden_layers[k] 
                                                + self.h_eta[k]
                                                * np.array(hidden_d[k])[:, None] @ np.array(y[k-1])[:, None].T
                                                #+ mom * (self.hidden_layers[k] - old_hidden_layers[k])
                                            )
                    old_hidden_layers[k] = aux_hidden_layer

    def test(self, x_test, y_test):
        success_sum = 0
        for i in range(0, len(x_test)):
            #Hidden layer neuron's activation
            u = []
            y = []

            #Calculates hidden layers
            for k in range(0, self.n_h_layers):
                if k == 0:
                    u_k = self.hidden_layers[k] @ (np.insert(x_test[i], 0, 1))
                else:
                    u_k = self.hidden_layers[k] @ y[k-1]
 
                u.append(u_k)
                
                y_k = []
                for l in range(0, len(u_k)):
                    y_k.append(self.__activation(u_k[l]))

                y.append(y_k)

            hidden_layer_output = y[len(self.hidden_layers) - 1]

            #Computes output layer
            output = self.output_layer @ np.array(np.insert(hidden_layer_output, 0, 1)).T
            for k in range(0, len(output)):
                output[k] = self.__activation(output[k])

            closest = 0
            for j in range(0, len(output)):
                if output[j] > output[closest]:
                    closest = j

            label = y_test[i]

            if label[closest] == 1:
                success_sum += 1

        self.results.append(100 * success_sum/len(x_test))

    def test_regression(self, x_test, y_test):
        success_sum = 0
        y_predicted = []
        for i in range(0, len(x_test)):
            #Hidden layer neuron's activation
            u = []
            y = []

            #Calculates hidden layers
            for k in range(0, self.n_h_layers):
                if k == 0:
                    u_k = self.hidden_layers[k] @ (np.insert(x_test[i], 0, 1))
                else:
                    u_k = self.hidden_layers[k] @ y[k-1]
 
                u.append(u_k)
                
                y_k = []
                for l in range(0, len(u_k)):
                    y_k.append(self.__activation(u_k[l]))

                y.append(y_k)

            hidden_layer_output = y[len(self.hidden_layers) - 1]

            #Computes output layer
            output = self.output_layer @ np.array(np.insert(hidden_layer_output, 0, 1)).T

            for k in range(0, len(output)):
                output[k] = self.__activation(output[k])
            
            y_predicted.append(output)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, len(y_test), 1))
        #ax.set_yticks(np.arange(-1, 1., 0.1))
        #plt.scatter(x, y)
        plt.grid()
        plt.plot([i for i in range(len(y_test))], y_test, 'r+')
        plt.plot([i for i in range(len(y_test))], y_predicted, 'bo')
        plt.show()


    def __str__(self):
        return "Mom: " + str(self.mom) + "; Epochs: " + str(self.epochs) + "; Layers: " + str(self.n_h_neurons + [self.n_o_neurons]) + "; Eta: " + str(self.h_eta + [self.eta]) + ""


n_rounds = 10
epochs = 10
learning_rate = 0.051181
mom = 0.75

sucess_rate_sum = 0

highest_sucess_rate = 0
lowest_sucess_rate = 1

x, y = utils.load_data_set()

n_training_samples = int(np.floor(0.8 * len(x)))
n_tests = int(np.floor(0.2 * len(x))) 

x_training = x[0:n_training_samples]
y_training = y[0:n_training_samples]

tests = [x[-n_tests:], y[-n_tests:]]

net = mlp(x_training, y_training)

net.config(epochs, learning_rate, mom, False, len(y_training[0]), 1, [50], [0.04214523])
net.train(n_training_samples)
net.test_regression([tests[0][i] for i in range(0, n_tests)], [tests[1][i] for i in range(0, n_tests)])
