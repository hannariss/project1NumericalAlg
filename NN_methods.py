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

    

