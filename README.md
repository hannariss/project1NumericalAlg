# project1NumericalAlg

**Authors:** Johanna Rissbacher, Jule Grimm

**Note:**


## Project Structure

The project contains the following files:

- **load_data.py**: contains functions to load the MNIST data and process it
- **NN_methods.py**: class FNN containing functions for the Neural Network
- **main.py**: executes the code for loading the data, training the NN and performing an attack

## Class Structure (NN_methods.py)
-**FNN**

## Class Description

### FNN:
- **Inputs**:
    - layer_sizes: each entry represents one layer and defines its size

- **Methods**:
    -`sigmoid`: Sigmoid activation function
    -`sigmoid_prime`: derivative of the sigmoid function
    -`feedforward`: Computes output of neural network (prediction) by iteratively calculating outputs of each layer
    -`SGD`: Stochastic gradient method for FNN
    -`backprop`: Perform backpropagation to approximate gradient of the loss function
    -`test_accuracy`: Return the number of test inputs for which the neural network outputs the correct result
    -`fgsm_attack`: Perform FGSM attack to generate an input with perturbation

## Description of load_data.py:

-**Methods**:
    -`process_data`: Convert original data into a data structure more convienient for NN
    internally used in `process_data`:
    -`load_data`: load provided MNIST data
    -`output_vector`: reshapes the provided y-values into a suitable array for the NN




