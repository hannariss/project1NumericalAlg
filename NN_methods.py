import numpy as np
import random

class FNN():
    def __init__(self, layer_sizes):
        """
        Create feedforward neural netword
        
        Parameters:
            layer_sizes: np.array, each entry represents one layer and defines its size
        """
        self.layer_sizes = layer_sizes
        self.T = len(layer_sizes) - 1
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1]+np.array(1), layer_sizes[1:])] #weights creates an array containing two arrays: weights for V0 to V1 and weights for V1 to V2
    
    def sigmoid(self, x):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_prime(self, x):
        """Derivative of the sigmoid function."""
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self, x_0):
        """
        Computes output of neural network (prediction) by iteratively calculating outputs of each layer

        Parameters:
            x_0: np.array
                input to the neural network 

        Return: 
            out: np.array
                predicted label corresponding to given input
        """
        out = x_0
        for w in self.weights:
            input = np.append(out, [1])
            a = np.dot(w, input)
            out = self.sigmoid(a)
        return out
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Stochastic gradient method for FNN

        Parameters:
            training_data: list containing np.arrays
                training data loaded and processed by the load_data.py file
            epochs: int
                number of epochs to train for
            mini_batch_size: int
                size of mini-batches
            eta: float
                learning rate
            test_data: list containing np.arrays
                optional input, if provided then the the algorithm will evaluate after each epoch the training progress

        """
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                grad_initialise = [np.zeros(w.shape) for w in self.weights] #initialise array with shape w for later addition of every gradient
                for x, y in mini_batch:
                    gradient_xy = self.backprop(x, y)[0]
                    gradient_sum = [grad_init+grad_xy for grad_init, grad_xy in zip(grad_initialise, gradient_xy)] #addition of single gradients
                self.weights = [w-(eta/len(mini_batch))*g for w, g in zip(self.weights, gradient_sum)]
            if test_data:
                n_test = len(test_data)
                correct_predictions, loss = self.test_accuracy(test_data)
                print("Epoch {0}: {1} / {2} correct predictions. Loss = {3}".format(j + 1, correct_predictions, n_test, np.round(loss, decimals=3)))
            else:
                print("Epoch {0} complete".format(j + 1))
    
    def backprop(self, x, y):
        """
        Perform backpropagation to approximate gradient of the loss function.

        Parameters:
            x: np.array
                input to the neural network 
            y: np.array
                true expected output

        Return:
            grad: np.array
                contains matrices of gradients with respect to the weights of the network (used in SGD)
            grad_input: np.array
                contains gradients with respect to input neurons (used in attack)
        """
        grad = [np.zeros(w.shape) for w in self.weights] # initialize array for gradients in shape of weights array
        input = np.append(x, [1])                        # append bias node
        a_vecs = []                                      # store activations of each layer
        outs = [input]                                   # store outputs of each layer
        l = 1                                            # current layer
        
        # Forward
        # Calculate all activations and outputs of each layer and store them in lists
        for w in self.weights:
            l += 1
            a = np.dot(w, input)
            a_vecs.append(a)
            out = self.sigmoid(a)    
            if not l == len(self.layer_sizes): # If we're not in the last layer, append 1 (bias)
                out = np.append(out, [1])    
            outs.append(out)
            input = out
        
        # Backward
        # Calculate delta for each layer and from that, the gradients with respect to the weights (grad)
        delta = outs[-1] - y
        grad[-1] = np.outer((delta * self.sigmoid_prime(a_vecs[-1])), outs[-2]) 
        for l in range(1, self.T):
            a = a_vecs[-l]
            sp = self.sigmoid_prime(a)
            delta = np.dot(self.weights[-l].T, (delta * sp))[:-1] # Ignore bias node 
            grad[-l-1] = np.outer((delta * self.sigmoid_prime(a_vecs[-l-1])), outs[-l-2])
         
        # Calculate gradient with respect to input (x)
        grad_input = np.dot(self.weights[0].T, (delta * self.sigmoid_prime(a_vecs[0])))[:-1] 
        
        return grad, grad_input
    
    def test_accuracy(self, test_data, attack=False, epsilon=0.1):
        """
        Return the number of test inputs for which the neural network
        outputs the correct result as well as the loss calculated by 
        using the quadratic loss function.
        
        Paramters:
            test_data: List of tuples (x, y) where
                    - x is the input to the network
                    - y is the actual expected output (label)
            attack: boolean
                set true when accuracy is tested after attack 
            epsilon: float
                Perturbation magnitude (value range: [0, 1])
        
        Returns:
            The count of correct predictions made by the network and the loss.
        """
        correct_predictions = 0  # Initialize counter for correct classifications
        loss = 0
        # Process each input (x) and label (y) in the test dataset
        for (x, y) in test_data:
            if attack:
                x = self.fgsm_attack(x, y, epsilon) # Use perturbed x as input if we perform an attack
            
            output = self.feedforward(x)
            predicted_number = np.argmax(output) 
            if predicted_number == np.argmax(y): # Check if the prediction matches the actual label
                correct_predictions += 1
            loss += np.linalg.norm(output - y)**2

        loss = 1 / (2 * len(test_data)) * loss   
        return correct_predictions, loss
    
    def fgsm_attack(self, x, y, epsilon):
        """
        Perform FGSM attack to generate an input with perturbation.
        
        Parameters:
            x: np.array
                input to the neural network 
            y: np.array
                true expected output
            epsilon: float
                Perturbation magnitude (value range: [0, 1])
        
        Returns:
            Perturbed input.
        """
        # Perform backpropagation to compute the gradient of the loss with respect to the input
        grad = self.backprop(x, y)[1] 
        perturbed_x = x + epsilon * np.sign(grad) # Generate the perturbation using the sign of the input gradient
        
        # Clip values to ensure the input data is within valid range (e.g., 0-1 for image pixels)
        perturbed_x = np.clip(perturbed_x, 0, 1)
        
        return perturbed_x
   