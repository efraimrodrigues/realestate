import csv
import numpy as np

def load_data_set():
    x = []
    y = []

    with open('data/data_set.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x.append(np.array(list(map(float, row[0:5]))))
            y.append([float(row[6])])

    x = np.array(x)
    y = np.array(y)

    for i in range(0, len(x[0])):
        x_max = max(np.array(x)[:,i])
        x_min = min(np.array(x)[:,i])

        x[:,i] = 2 * (np.array(np.array(x)[:,i]) - x_min)/(x_max - x_min) - 1

    y_max = max(y)
    y_min = min(y)

    y = 2 * (y - y_min)/(y_max - y_min) - 1
    
    return x, y
