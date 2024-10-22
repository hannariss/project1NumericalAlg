import NN_methods
import load_data

# Load data
training_data, validation_data, test_data = load_data.process_data()

# Train network
network = NN_methods.FNN([784, 30, 10])
network.SGD(training_data, 15, 10, 3.0, test_data=test_data)

# Attack
correct_on_attack = network.test_accuracy(test_data, attack=True, epsilon=0.01)
print(f"\nAfter attack: {correct_on_attack} / {len(test_data)} correct predictions")