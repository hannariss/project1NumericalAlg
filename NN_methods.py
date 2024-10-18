import numpy as np
import random

class FNN():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.T = 2
    
    def iteration(self, x_0):
        out = x_0
        for w in self.weights:
            input = np.append(out, [1])
            a = np.dot(w, input)
            out = self.sigmoid(a)
        return out

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def backprop(self, x, y, weights):
        grad = [np.zeros(w.shape) for w in weights]

        # forward
        input = np.append(x, [1])
        a_vecs = [] # store activations of each layer
        outs = [x] # store outputs of each layer
        l = 1 # current layer
        for w in weights:
            l += 1
            a = np.dot(w, input)
            a_vecs.append(a)
            out = self.sigma(a)
            if not l == len(self.layer_sizes): # If we're not in the last layer, append 1 (bias)
                out = np.append(out, [1])
            outs.append(out)
            input = out
        # backward
        delta = outs[-1] - y
        grad[-1] = np.outer((delta * self.sigmoid_prime(a_vecs[-1])), outs[-2])
        for l in range(1, self.T):
            a = a_vecs[-l]
            sp = self.sigmoid_prime(a)
            delta = np.dot(weights[-l].T, delta * sp)
            grad[-l-1] = np.outer((delta * self.sigmoid_prime(a_vecs[-l-1])), outs[-l-2])
        return grad

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network
        outputs the correct result.
        
        Args:
            test_data: List of tuples (x, y) where
                    - x is the input to the network
                    - y is the actual expected output (label)
        
        Returns:
            The count of correct predictions made by the network.
        """
        correct_predictions = 0  # Initialize counter for correct classifications
        
        # Process each input (x) and label (y) in the test dataset
        for (x, y) in test_data:
            output_activations = self.iteration(x)
            predicted_number = np.argmax(output_activations)
            if predicted_number == y: # Check if the prediction matches the actual label
                correct_predictions += 1  # Increment counter if correct
        
        return correct_predictions