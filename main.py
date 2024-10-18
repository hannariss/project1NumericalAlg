import NN_methods
import load_data

training_data, validation_data, test_data = load_data.data_control()

network = NN_methods.FNN([784, 30, 10])
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)