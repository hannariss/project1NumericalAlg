import pickle
import gzip
import numpy as np
import os

PATH_DATA = os.path.join('.', 'data', 'mnist.pkl.gz')

def load_data(path = PATH_DATA):
    f = gzip.open(path)
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def data_control():
    """
    Convert original data into a data structure more convienient for NN
    training_data consists of a tuple (x, y), x containing 50.000 entires with each 784 values (28x28 pixels)
    -> reshape each entry in x to an array(784,1)
    y gets turned into (10,1) array filled with zero, except on the position of the number it represents (there will be placed a 1) 
    """
    tr, val, test = load_data()

    training_x = [np.reshape(x, (784, 1)) for x in tr[0]]
    training_y = [training_output_vector(y) for y in tr[1]]
    training_data = zip(training_x, training_y)

    validation_x = [np.reshape(x, (784, 1)) for x in val[0]]
    validation_y = [training_output_vector(y) for y in val[1]]
    validation_data = zip(validation_x, validation_y)

    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [training_output_vector(y) for y in test[1]]
    test_data = zip(test_x, test_y)
    return (training_data, validation_data, test_data)

def training_output_vector(i):
    """returns an array with shape (10, 1)
    zeros on all entries except at the postion of the number represented by x
    At this position (i) a 1 will be placed

    Example
    --------
    >>> training_output_vector(4)
    array([[0.],
           [0.],
           [0.],
           [0.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.]])
    """
    a = np.zeros((10, 1))
    a[i] = 1.0
    return a
