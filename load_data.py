import pickle
import gzip
import numpy as np
import os

PATH_DATA = os.path.join('.', 'data', 'mnist.pkl.gz')

def load_data(path = PATH_DATA):
    """
    load provided MNIST data
    """
    f = gzip.open(path)
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def data_control():
    """
    Convert original data into a data structure more convienient for NN
    eg. Training_data consists of a tuple (np.array(x), np.array(y)), x containing 50.000 entries (arrays) with each 784 values (28x28 pixels)
    -> reshape each array in x to an array(784, )
    y gets turned into (10, ) array filled with zero, except on the position of the number it represents (there will be placed a 1)
    (this is achieved by usining the function training_output_vector)

    Returns:
    out: list containing tuples of two arrays for x and y, list((np.array(784,), np.array(10,)),....)
        each for training, test and validation data
    """
    tr, val, test = load_data()

    # Reshape training data x and y
    training_x = [np.reshape(x, (784,)) for x in tr[0]]  # Flatten to 1D array (784,)
    training_y = [training_output_vector(y) for y in tr[1]]  # Output is 1D array (10,)
    training_data = list(zip(training_x, training_y))

    # Reshape validation data x and y
    validation_x = [np.reshape(x, (784,)) for x in val[0]]  # Flatten to 1D array (784,)
    validation_y = [training_output_vector(y) for y in val[1]]  # Output is 1D array (10,)
    validation_data = list(zip(validation_x, validation_y))

    # Reshape test data x and y
    test_x = [np.reshape(x, (784,)) for x in test[0]]  # Flatten to 1D array (784,)
    test_y = [training_output_vector(y) for y in test[1]]  # Output is 1D array (10,)
    test_data = list(zip(test_x, test_y))
    
    return (training_data, validation_data, test_data)


def training_output_vector(i):
    """
    Returns a 1D array with shape (10,)
    Zeros on all entries except at the position of the number represented by i.
    At this position (i), a 1 will be placed.

    Example
    --------
    >>> training_output_vector(4)
    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    """
    a = np.zeros(10)  # Create a 1D array of 10 zeros
    a[i] = 1.0        # Set the i-th index to 1.0
    return a