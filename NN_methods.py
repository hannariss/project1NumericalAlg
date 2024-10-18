import numpy as np

class FNN():
    def __init__(self, num_nodes0, num_nodes1, num_nodes2, sigma):
        self.num_nodes0 = num_nodes0
        self.num_nodes1 = num_nodes1
        self.num_nodes2 = num_nodes2
        self.T = 2
        self.sigma = sigma

    def input(self, w, x):
        """
        Parameters:
        w: np.array
            Matrix shaped |V_(t-1)| x |V_t|
        x: np.array
            input vector

        Out:
        a: np.array
            array containing input for a layer (|V_(t)|)
        """
        a = np.dot(w.T, x)
        return a
    
    def iteration(self, x_0):
        w1 = np.zeros(self.num_nodes0, self.num_nodes1)
        w2 = np.zeros(self.num_nodes1, self.num_nodes2)
        a_1 = self.input(w1, x_0)
        o_1 = self.sigma(a_1)
        a_2 = self.input(w2, o_1)
        output_FNN = self.sigma(a_2)
        return output_FNN

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                grad_initialise = [np.zeros(w.shape) for w in self.weights] #initialise array with shape w for later addition of every gradient
                for x, y in mini_batch:
                    gradient_xy = self.backprop(x, y)
                    gradient_sum = [grad_init+grad_xy for grad_init, grad_xy in zip(grad_initialise, gradient_xy)] #addition of single gradients
                self.weights = [w-(eta/len(mini_batch))*g for w, g in zip(self.weights, gradient_sum)]
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
    

